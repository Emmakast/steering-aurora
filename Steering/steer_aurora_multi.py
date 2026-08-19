import sys
import os
import argparse
import torch
import pandas as pd
import gc
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

import tempfile


try:
    import boto3
    from dotenv import load_dotenv
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Add the steering script directory to path to import helpers
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/scripts/steering') # Original
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/steering/scripts') # Parent scripts dir
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/steering')
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader') # Exact location of extract_latents_hres
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # robust fallback

try:
    from extract_latents_hres import prepare_batch, batch_to_dataset, download_data, download_static
except ImportError as e:
    print(f"Warning: Could not import prepare_batch and batch_to_dataset from extract_latents_hres.py: {e}")
    sys.exit(1)

try:
    from aurora import Aurora, rollout
except ImportError:
    print("Warning: Could not import Aurora. Make sure the aurora environment is active.")

def build_polar_lat_mask(lat_size: int, lat_min: float = 60.0, hemisphere: str = "both") -> torch.Tensor:
    """Create a 1D boolean mask over latent latitude rows."""
    latitudes = torch.linspace(90.0, -90.0, steps=lat_size)
    if hemisphere == "north":
        return latitudes >= lat_min
    if hemisphere == "south":
        return latitudes <= -lat_min
    return latitudes.abs() >= lat_min


def apply_spatial_mask_to_delta(
    delta_v: torch.Tensor,
    mask_region: str = "polar",
    polar_lat_min: float = 60.0,
    hemisphere: str = "both",
) -> torch.Tensor:
    """
    Apply spatial mask to steering vector.
    Expected latent shape for Aurora encoder: [1, 16200, C] (90 lat x 180 lon)
    """
    if mask_region == "none":
        return delta_v

    if delta_v.ndim != 3:
        print(f"Warning: unexpected delta_v ndim={delta_v.ndim}; skipping spatial mask.")
        return delta_v

    _, seq_len, _ = delta_v.shape
    
    if seq_len not in [16200, 64800]:
        print(f"Warning: could not apply spatial mask for seq_len={seq_len}. Skipping mask for this layer.")
        return delta_v

    if mask_region == "polar":
        if seq_len == 16200:
            lat_size = 90
            lon_size = 180
        else:
            lat_size = 180
            lon_size = 360
            
        lat_mask_1d = build_polar_lat_mask(lat_size, lat_min=polar_lat_min, hemisphere=hemisphere)
        # Expand to lat x lon, then flatten to match token sequence
        spatial_mask = lat_mask_1d.unsqueeze(1).expand(lat_size, lon_size).reshape(1, seq_len, 1)
        spatial_mask = spatial_mask.to(dtype=delta_v.dtype, device=delta_v.device)
        return delta_v * spatial_mask

    print(f"Warning: could not apply mask for seq_len={seq_len}, skipping mask.")
    return delta_v


def main():
    parser = argparse.ArgumentParser(description="Contrastive Activation Addition Steering")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0], help="List of steering strengths")
    parser.add_argument("--layer-weights", type=float, nargs="+", default=[0.1, 0.3, 1.0], help="Layer-specific alpha weight factors (e.g. for encoder_0, encoder_1, encoder_2)")
    parser.add_argument("--phenomenon", type=str, default="AO", choices=["AO", "MJO", "ENSO", "AAO"], help="Phenomenon to steer")
    parser.add_argument("--csv", type=str, default="target_dates.csv", help="Target dates CSV file")
    parser.add_argument("--neutral-csv", type=str, default=None, help="Optional separate CSV file for Neutral dates")
    parser.add_argument("--name-suffix", type=str, default="", help="Suffix to append to the output filename (e.g., '_ao81')")
    parser.add_argument("--steps", type=int, default=12, help="Number of rollout steps (12 = 3 days at 6-hour steps)")
    parser.add_argument("--init-hour", type=int, default=12, choices=[0, 12], help="Initialization hour for inference (0 or 12)")
    parser.add_argument("--base-date", type=str, default=None, help="Optional fixed base date YYYY-MM-DD (otherwise first neutral date)")
    parser.add_argument("--mask-region", type=str, default="polar", choices=["none", "polar"], help="Spatial mask region for steering vector")
    parser.add_argument("--polar-lat-min", type=float, default=60.0, help="Polar mask starts at |lat| >= this value")
    parser.add_argument("--hemisphere", type=str, default="both", choices=["both", "north", "south"], help="Polar mask hemisphere")
    parser.add_argument("--out-dir", type=str, default="thesis/results", help="Output directory for vectors and predictions")
    args = parser.parse_args()
    
    suffix_str = args.name_suffix if args.name_suffix.startswith("_") or args.name_suffix == "" else f"_{args.name_suffix}"
    print(f"Starting Contrastive Activation Addition (CAA) Steering Pipeline ({args.phenomenon}, alphas={args.alphas})...")
    
    # ==========================================
    # Step 1: Compute the Steering Vector
    # ==========================================
    csv_path = args.csv
    latents_dir = Path("/tmp/ekasteleyn/aurora_hres_latents") # Update this if needed, user didn't specify exactly
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Filter for the chosen phenomenon
    phenom_df = df[df['Phenomenon'] == args.phenomenon]
    active_dates = phenom_df[phenom_df['Type'] == 'Active']
    
    if args.neutral_csv and os.path.exists(args.neutral_csv):
        neutral_df = pd.read_csv(args.neutral_csv)
        neutral_phenom_df = neutral_df[neutral_df['Phenomenon'] == args.phenomenon]
        neutral_dates = neutral_phenom_df[neutral_phenom_df['Type'] == 'Neutral']
        print(f"Loaded Neutral dates from separate CSV: {args.neutral_csv}")
    else:
        neutral_dates = phenom_df[phenom_df['Type'] == 'Neutral']
    
    print(f"Loaded CSV: {csv_path}")
    print(f"Found {len(active_dates)} Active dates and {len(neutral_dates)} Neutral dates.")
    
# S3 Client setup
    s3_client = None
    if HAS_BOTO3:
        load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
        access_key = os.getenv('UVA_S3_ACCESS_KEY')
        secret_key = os.getenv('UVA_S3_SECRET_KEY')
        if access_key and secret_key:
            s3_client = boto3.client(
                's3',
                endpoint_url="https://ceph-gw.science.uva.nl:8000",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print("S3 Client initialized.")
            
    def load_mean_latent(dates_df, layer='encoder_2', hhmm='0000'):
        tensors = []
        for _, row in dates_df.iterrows():
            date_str = f"{int(row['Year']):04d}{int(row['Month']):02d}{int(row['Day']):02d}"
            filename = f"latent_{date_str}_{hhmm}_{layer}.pt"
            
            # Check local first
            possible_paths = [
                Path(filename),
                Path("thesis/results") / filename,
                Path(os.environ.get('TMPDIR', '/tmp/ekasteleyn')) / "aurora_hres_latents" / filename
            ]
            
            file_path = None
            for p in possible_paths:
                if p.exists():
                    file_path = p
                    break
                    
            if file_path is None and s3_client is not None:
                # Try downloading from S3
                s3_key = f"aurora_hres_validation/{filename}"
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        s3_client.download_file("ekasteleyn-aurora-predictions", s3_key, tmp.name)
                        file_path = Path(tmp.name)
                except Exception as e:
                    # print(f"Could not download {filename} from S3: {e}")
                    pass
                    
            if file_path is None or not file_path.exists():
                print(f"Warning: Latent file {filename} not found locally or on S3.")
                continue
                
            try:
                t = torch.load(file_path, weights_only=True, map_location='cpu').float()
                tensors.append(t)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
            if str(file_path).startswith('/tmp') and 'tmp' in file_path.name:
                file_path.unlink() # cleanup named temp file
            
        if not tensors:
            print(f"Warning: No valid latent files found for {dates_df['Type'].iloc[0]}. Returning empty.")
            return None
            
        stacked = torch.stack(tensors, dim=0)
        # Use nanmean to ignore NaNs in the dataset, and zero out any fully-NaN results
        mean_val = torch.nanmean(stacked, dim=0)
        mean_val = torch.nan_to_num(mean_val, nan=0.0, posinf=0.0, neginf=0.0)
        return mean_val

        
    steering_vectors = {}
    layer_names = ['encoder_0', 'encoder_1', 'encoder_2']
    
    for layer in layer_names:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        steering_vec_path = output_dir / f"steering_vector_{args.phenomenon.lower()}_{layer}{suffix_str}.pt"
        steering_norm_path = output_dir / f"steering_vector_norm_{args.phenomenon.lower()}_{layer}{suffix_str}.pt"
        
        if steering_vec_path.exists():
            print(f"✓ Found PRE-COMPUTED steering vector! Loading {steering_vec_path.name} directly...")
            raw_vec = torch.load(steering_vec_path, map_location='cpu', weights_only=True)
        else:
            print(f"Loading Active latents for {layer} ({len(active_dates)} dates)...")
            mean_active = load_mean_latent(active_dates, layer=layer)
            
            print(f"Loading Neutral latents for {layer} ({len(neutral_dates)} dates)...")
            mean_neutral = load_mean_latent(neutral_dates, layer=layer)
            
            if mean_active is None or mean_neutral is None:
                print(f"Missing latents for {layer}. Creating dummy delta_v.")
                delta_v = torch.zeros((1, 4, 18, 36, 1536)) if layer != 'encoder_0' else torch.zeros((1, 18, 36, 1536))
            else:
                delta_v = mean_active - mean_neutral
                delta_v = torch.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0)
    
            print(f"{layer} Steering vector (delta_v) shape: {delta_v.shape}, max={torch.max(delta_v)}, min={torch.min(delta_v)}, norm={torch.norm(delta_v)}")
            raw_vec = delta_v
            
            # Save unmasked plot-ready tensors before applying mask
            torch.save(raw_vec.cpu(), steering_vec_path)
            torch.save(torch.norm(raw_vec, dim=-1).squeeze(0).cpu(), steering_norm_path)
            print(f"Saved {layer} steering vector to {steering_vec_path.name}")
            print(f"Saved {layer} steering norm to {steering_norm_path.name}")

        # Always apply user-selected spatial mask on the fly
        masked_delta_v = apply_spatial_mask_to_delta(
            raw_vec,
            mask_region=args.mask_region,
            polar_lat_min=args.polar_lat_min,
            hemisphere=args.hemisphere,
        )
        nz_ratio = (masked_delta_v != 0).float().mean().item()
        print(
            f"Applied on-the-fly mask to {layer}: region={args.mask_region}, hemisphere={args.hemisphere}, "
            f"lat_min={args.polar_lat_min}. Non-zero fraction={nz_ratio:.4f}"
        )

        steering_vectors[layer] = masked_delta_v
        
    # ==========================================
    # Step 2: Implement the Intervention Hook (Normalized)
    # ==========================================
    def make_intervention_hook(steering_vec, alpha=1.0):
        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            
            # Move vector to current device and dtype
            s_vec = steering_vec.to(dtype=x.dtype, device=x.device)
            
            # Standard CAA: Directly inject the scaled difference vector
            # This preserves the physical spatial gradients extracted from your CSV dates
            new_x = x + (alpha * s_vec)
            
            if is_tuple:
                return (new_x,) + output[1:]
            return new_x
        return hook
        
    # ==========================================
    # Step 3: Prepare the Data
    # ==========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prioritize persistent scratch-shared so we don't redownload data on every new job
    shared_scratch = Path("/scratch-shared/ekasteleyn/aurora_data")
    if shared_scratch.parent.exists():
        download_dir = shared_scratch
    else:
        # Fallback to local job scratch node
        download_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "aurora_data"
        
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {download_dir} for data...")
    print("Downloading static/base data if needed...")
    download_static(download_dir)

    # Select base date
    if args.base_date:
        base_day_str = args.base_date
    else:
        base_date = neutral_dates.iloc[0]
        base_day_str = f"{int(base_date['Year']):04d}-{int(base_date['Month']):02d}-{int(base_date['Day']):02d}"
    print(f"Selected Base Date: {base_day_str}")

    download_data(base_day_str, download_dir)
    if args.init_hour == 0:
        prev_day_str = (pd.to_datetime(base_day_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        download_data(prev_day_str, download_dir)

    print("Preparing batch...")
    batch = prepare_batch(base_day_str, download_dir, init_hour=args.init_hour)
    date_tag = base_day_str.replace("-", "")
    init_tag = f"{args.init_hour:02d}00"

    # ==========================================
    # Step 4: Run the Steered Inference
    # ==========================================
    print(f"Loading Aurora Model on {device}...")
    
    # Needs Aurora installed
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)

    # Move to device (handle tuples or direct tensors)
    if isinstance(batch, tuple):
        batch = tuple(t.to(device) if hasattr(t, 'to') else t for t in batch)
    else:
        batch = batch.to(device)
    
    base_output_filename = os.path.join(args.out_dir, f"base_{args.phenomenon.lower()}{suffix_str}_{date_tag}_{init_tag}_alpha_0.0.nc")
    
    if not os.path.exists(base_output_filename):
        print(f"Running base inference (alpha=0.0) without hook for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                base_pred_batch = pred
                
        base_pred_batch = base_pred_batch.to("cpu")
        base_ds = batch_to_dataset(base_pred_batch, step=args.steps)
        
        tmp_base_filename = f"{base_output_filename}.tmp_base"
        base_ds.to_netcdf(tmp_base_filename)
        os.rename(tmp_base_filename, base_output_filename)
        print(f"Saved base output to {base_output_filename}")
    else:
        print(f"Base output {base_output_filename} already exists, skipping base inference.")
        
    for alpha_val in args.alphas:
        print(f"Applying hooks with base alpha={alpha_val}...")

        hook_handles = []
        for idx, (layer_name, weight) in enumerate(zip(layer_names, args.layer_weights)):
            scaled_alpha = alpha_val * weight
            print(f"Registering hook for {layer_name} with alpha={scaled_alpha} (weight={weight})")
            
            s_vec = steering_vectors[layer_name]
            
            handle = model.backbone.encoder_layers[idx].register_forward_hook(
                make_intervention_hook(s_vec, alpha=scaled_alpha)
            )
            hook_handles.append(handle)

        print(f"Running steered inference (base alpha={alpha_val}) for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                pred_batch = pred

        pred_batch = pred_batch.to("cpu")
        
        for handle in hook_handles:
            handle.remove()

        print(f"Converting prediction to xarray for alpha={alpha_val}...")
        ds = batch_to_dataset(pred_batch, step=args.steps)

        lat_tag = str(args.polar_lat_min).replace(".", "p")
        mask_tag = "nomask" if args.mask_region == "none" else f"polar_{args.hemisphere}_lat{lat_tag}"
        output_filename = os.path.join(
            args.out_dir,
            f"steered_{args.phenomenon.lower()}{suffix_str}_{date_tag}_{init_tag}_"
            f"{mask_tag}_alpha_{alpha_val}.nc"
        )
        ds.to_netcdf(output_filename)
        print(f"Saved steered output to {output_filename}")

    print("Done!")

if __name__ == "__main__":
    main()