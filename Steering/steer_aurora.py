import sys
import os

# Disable HDF5 file locking to prevent "NetCDF: HDF error" on GPFS via xarray/dask when running multiple jobs
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
sys.path.append('/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader')
try:
    from extract_latents_hres import prepare_batch, batch_to_dataset, download_data, download_static
except ImportError:
    print("Warning: Could not import prepare_batch and batch_to_dataset from extract_latents_hres.py")
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

def build_tropical_lat_mask(lat_size: int, lat_max: float = 30.0) -> torch.Tensor:
    """Create a 1D boolean mask over latent latitude rows for the tropics."""
    latitudes = torch.linspace(90.0, -90.0, steps=lat_size)
    return latitudes.abs() <= lat_max

def apply_spatial_mask_to_delta(
    delta_v: torch.Tensor,
    mask_region: str = "polar",
    polar_lat_min: float = 60.0,
    hemisphere: str = "both",
    tropical_lat_max: float = 30.0,
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
    # Aurora uses 90 lat x 180 lon = 16200 tokens (for encoder 1 and 2)
    # and 180 lat x 360 lon = 64800 tokens (for encoder 0)
    if seq_len in [16200, 64800] and mask_region in ["polar", "tropical"]:
        if seq_len == 16200:
            lat_size = 90
            lon_size = 180
        else:
            lat_size = 180
            lon_size = 360
            
        if mask_region == "polar":
            lat_mask_1d = build_polar_lat_mask(lat_size, lat_min=polar_lat_min, hemisphere=hemisphere)
        elif mask_region == "tropical":
            lat_mask_1d = build_tropical_lat_mask(lat_size, lat_max=tropical_lat_max)
            
        # Expand to lat x lon, then flatten to match token sequence
        spatial_mask = lat_mask_1d.unsqueeze(1).expand(lat_size, lon_size).reshape(1, seq_len, 1)
        spatial_mask = spatial_mask.to(dtype=delta_v.dtype, device=delta_v.device)
        return delta_v * spatial_mask

    print(f"Warning: could not apply mask for seq_len={seq_len}, skipping mask.")
    return delta_v


def main():
    parser = argparse.ArgumentParser(description="Contrastive Activation Addition Steering")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0], help="List of steering strengths")
    parser.add_argument("--phenomenon", type=str, default="AO", choices=["AO", "MJO", "ENSO", "AAO", "NAO", "PNA"], help="Phenomenon to steer")
    parser.add_argument("--csv", type=str, default="target_dates.csv", help="Target dates CSV file")
    parser.add_argument("--neutral-csv", type=str, default=None, help="Optional separate CSV file for Neutral dates")
    parser.add_argument("--name-suffix", type=str, default="", help="Suffix to append to the output filename (e.g., '_ao81')")
    parser.add_argument("--steps", type=int, default=12, help="Number of rollout steps (12 = 3 days at 6-hour steps)")
    parser.add_argument("--init-hour", type=int, default=12, choices=[0, 12], help="Initialization hour for inference (0 or 12)")
    parser.add_argument("--base-date", type=str, default=None, help="Optional fixed base date YYYY-MM-DD (otherwise first neutral date)")
    parser.add_argument("--mask-region", type=str, default="polar", choices=["none", "polar", "tropical"], help="Spatial mask region for steering vector")
    parser.add_argument("--polar-lat-min", type=float, default=60.0, help="Polar mask starts at |lat| >= this value")
    parser.add_argument("--tropical-lat-max", type=float, default=30.0, help="Tropical mask ends at |lat| <= this value")
    parser.add_argument("--hemisphere", type=str, default="both", choices=["both", "north", "south"], help="Polar mask hemisphere")
    parser.add_argument("--encoder-idx", type=int, default=2, help="Encoder index to steer (0, 1, or 2)")
    parser.add_argument("--output-dir", type=str, default="thesis/results", help="Directory to save the vectors and netcdf outputs")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory for downloaded HRES data (shared between scripts to avoid re-downloading)")
    parser.add_argument("--inject-once", action="store_true", help="Only inject the steering vector at the first rollout step")
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

    # Auto-convert contrastive pairs format to standard format
    if 'active_date' in df.columns and 'neutral_date' in df.columns:
        parsed_rows = []
        for _, r in df.iterrows():
            ad = pd.to_datetime(r['active_date'])
            nd = pd.to_datetime(r['neutral_date'])
            parsed_rows.append({'Year': ad.year, 'Month': ad.month, 'Day': ad.day, 'Type': 'Active', 'Phenomenon': args.phenomenon})
            parsed_rows.append({'Year': nd.year, 'Month': nd.month, 'Day': nd.day, 'Type': 'Neutral', 'Phenomenon': args.phenomenon})
        df = pd.DataFrame(parsed_rows)
    
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
    
    # Check if pre-computed steering vector already exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steering_vec_path = output_dir / f"steering_vector_{args.phenomenon.lower()}{suffix_str}.pt"
    steering_norm_path = output_dir / f"steering_vector_norm_{args.phenomenon.lower()}{suffix_str}.pt"

    if steering_vec_path.exists():
        print(f"✓ Found PRE-COMPUTED steering vector! Loading {steering_vec_path.name} directly...")
        steering_vec = torch.load(steering_vec_path, map_location='cpu', weights_only=True)
    else:
        print("Pre-computed steering vector not found. Computing from scratch...")
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
    
            
        print(f"Loading Active latents ({len(active_dates)} dates)...")
        mean_active = load_mean_latent(active_dates, layer=f'encoder_{args.encoder_idx}')
        
        print(f"Loading Neutral latents ({len(neutral_dates)} dates)...")
        mean_neutral = load_mean_latent(neutral_dates, layer=f'encoder_{args.encoder_idx}')
        
        if mean_active is None or mean_neutral is None:
            print("Missing latents. Creating dummy delta_v for demonstration purposes.")
            # Dummy shape for Aurora Swin: [1, 4, H, W, C]
            delta_v = torch.zeros((1, 4, 18, 36, 1536))
        else:
            delta_v = mean_active - mean_neutral
            delta_v = torch.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0)
    
        print(f"Steering vector (delta_v) shape: {delta_v.shape}, max={torch.max(delta_v)}, min={torch.min(delta_v)}, norm={torch.norm(delta_v)}")
    
    
        # Save unmasked plot-ready tensors before applying mask (if generating new)
        # Note: we save it unmasked so we can apply different masks on the fly later if needed
        # Or keep it as is. Actually, if we just compute delta_v, we can save it.
        steering_vec = delta_v

    
        # Save plot-ready tensors
        torch.save(steering_vec.cpu(), steering_vec_path)
        torch.save(torch.norm(steering_vec, dim=-1).squeeze(0).cpu(), steering_norm_path)
        print(f"Saved steering vector to {steering_vec_path.name}")
        print(f"Saved steering norm to {steering_norm_path.name}")
        
        if s3_client is not None:
            try:
                s3_key_vec = f"aurora_hres_validation/vectors/{steering_vec_path.name}"
                s3_client.upload_file(str(steering_vec_path), "ekasteleyn-aurora-predictions", s3_key_vec)
                print(f"      ↑ Uploaded steering vector to S3: s3://ekasteleyn-aurora-predictions/{s3_key_vec}")
                
                s3_key_norm = f"aurora_hres_validation/vectors/{steering_norm_path.name}"
                s3_client.upload_file(str(steering_norm_path), "ekasteleyn-aurora-predictions", s3_key_norm)
                print(f"      ↑ Uploaded steering norm to S3: s3://ekasteleyn-aurora-predictions/{s3_key_norm}")
            except Exception as e:
                print(f"      ⚠ Failed to upload steering vector to S3: {e}")
                
    # Always apply spatial mask on the fly (whether loaded or newly computed)
    steering_vec = apply_spatial_mask_to_delta(
        steering_vec,
        mask_region=args.mask_region,
        polar_lat_min=args.polar_lat_min,
        hemisphere=args.hemisphere,
        tropical_lat_max=args.tropical_lat_max,
    )
    nz_ratio = (steering_vec != 0).float().mean().item()
    lat_info = f"lat_min={args.polar_lat_min}" if args.mask_region == "polar" else f"lat_max={args.tropical_lat_max}" if args.mask_region == "tropical" else ""
    print(f"Applied on-the-fly mask: region={args.mask_region}, hemisphere={args.hemisphere}, {lat_info}. Non-zero fraction={nz_ratio:.4f}")
        
    # ==========================================
    # Step 2: Implement the Intervention Hook (Normalized)
    # ==========================================
    def make_intervention_hook(steering_vec, alpha=1.0, inject_once=False):
        state = {'count': 0}
        def hook(module, args, output):
            if inject_once and state['count'] >= 1:
                return output
            state['count'] += 1
            
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
    
    # Prioritize --data-dir argument if provided (to share data with extract_latents_hres.py)
    if args.data_dir:
        download_dir = Path(args.data_dir)
    else:
        # Fallback: prefer persistent scratch-shared, else SLURM TMPDIR
        shared_scratch = Path("/scratch-shared/ekasteleyn/aurora_data")
        if shared_scratch.parent.exists():
            download_dir = shared_scratch
        else:
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
    
    base_output_filename = f"base_{args.phenomenon.lower()}{suffix_str}_{date_tag}_{init_tag}_alpha_0.0.nc"
    base_output_path = output_dir / base_output_filename
    
    if not os.path.exists(base_output_path):
        print(f"Running base inference (alpha=0.0) without hook for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                base_pred_batch = pred
                
        base_pred_batch = base_pred_batch.to("cpu")
        base_ds = batch_to_dataset(base_pred_batch, step=args.steps)
        
        tmp_base_filename = output_dir / f"{base_output_filename}.tmp_base"
        base_ds.to_netcdf(tmp_base_filename)
        os.rename(tmp_base_filename, base_output_path)
        print(f"Saved base output to {base_output_path}")
    else:
        print(f"Base output {base_output_filename} already exists, skipping base inference.")
        
    for alpha_val in args.alphas:
        print(f"Applying hook with alpha={alpha_val}...")

        hook_handle = model.backbone.encoder_layers[args.encoder_idx].register_forward_hook(
            make_intervention_hook(steering_vec, alpha=alpha_val, inject_once=args.inject_once)
        )

        print(f"Running steered inference (alpha={alpha_val}) for {args.steps} steps...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                pred_batch = pred

        pred_batch = pred_batch.to("cpu")
        hook_handle.remove()

        print(f"Converting prediction to xarray for alpha={alpha_val}...")
        ds = batch_to_dataset(pred_batch, step=args.steps)

        if args.mask_region == "none":
            mask_tag = "nomask"
        elif args.mask_region == "tropical":
            lat_tag = str(args.tropical_lat_max).replace(".", "p")
            mask_tag = f"tropical_lat{lat_tag}"
        else:
            lat_tag = str(args.polar_lat_min).replace(".", "p")
            mask_tag = f"polar_{args.hemisphere}_lat{lat_tag}"
        output_filename = output_dir / (
            f"steered_{args.phenomenon.lower()}{suffix_str}_{date_tag}_{init_tag}_"
            f"{mask_tag}_alpha_{alpha_val}.nc"
        )
        ds.to_netcdf(output_filename)
        print(f"Saved steered output to {output_filename}")

    print("Done!")

if __name__ == "__main__":
    main()