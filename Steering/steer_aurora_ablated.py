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
    latitudes = torch.linspace(90.0, -90.0, steps=lat_size)
    if hemisphere == "north": return latitudes >= lat_min
    if hemisphere == "south": return latitudes <= -lat_min
    return latitudes.abs() >= lat_min


def apply_spatial_mask_to_delta(delta_v: torch.Tensor, mask_region: str = "polar", polar_lat_min: float = 60.0, hemisphere: str = "both") -> torch.Tensor:
    if mask_region == "none": return delta_v
    if delta_v.ndim != 3: return delta_v
    _, seq_len, _ = delta_v.shape
    if seq_len == 16200 and mask_region == "polar":
        lat_size, lon_size = 90, 180
        lat_mask_1d = build_polar_lat_mask(lat_size, lat_min=polar_lat_min, hemisphere=hemisphere)
        spatial_mask = lat_mask_1d.unsqueeze(1).expand(lat_size, lon_size).reshape(1, seq_len, 1)
        spatial_mask = spatial_mask.to(dtype=delta_v.dtype, device=delta_v.device)
        return delta_v * spatial_mask
    return delta_v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--phenomenon", type=str, default="AO")
    parser.add_argument("--csv", type=str, default="target_dates.csv")
    parser.add_argument("--neutral-csv", type=str, default=None)
    parser.add_argument("--name-suffix", type=str, default="")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--init-hour", type=int, default=12)
    parser.add_argument("--base-date", type=str, default=None)
    parser.add_argument("--mask-region", type=str, default="polar")
    parser.add_argument("--polar-lat-min", type=float, default=60.0)
    parser.add_argument("--hemisphere", type=str, default="both")
    parser.add_argument("--latent-idx", type=int, required=True, help="Latent (Z) index to steer (0, 1, or 2)")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the vectors and netcdf outputs")
    parser.add_argument("--data-dir", type=str, default=None, help="Shared download dir for HRES data (avoids re-downloading across jobs)")
    args = parser.parse_args()
    
    suffix_str = args.name_suffix if args.name_suffix.startswith("_") or args.name_suffix == "" else f"_{args.name_suffix}"
    
    csv_path = args.csv
    df = pd.read_csv(csv_path)
    phenom_df = df[df['Phenomenon'] == args.phenomenon]
    active_dates = phenom_df[phenom_df['Type'] == 'Active']
    neutral_dates = phenom_df[phenom_df['Type'] == 'Neutral']
    
    s3_client = None
    if HAS_BOTO3:
        load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
        access_key = os.getenv('UVA_S3_ACCESS_KEY')
        secret_key = os.getenv('UVA_S3_SECRET_KEY')
        if access_key and secret_key:
            s3_client = boto3.client('s3', endpoint_url="https://ceph-gw.science.uva.nl:8000", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
            
    def load_mean_latent(dates_df, layer='encoder_2', hhmm='0000'):
        tensors = []
        for _, row in dates_df.iterrows():
            date_str = f"{int(row['Year']):04d}{int(row['Month']):02d}{int(row['Day']):02d}"
            filename = f"latent_{date_str}_{hhmm}_{layer}.pt"
            possible_paths = [Path(filename), Path("thesis/results") / filename, Path(os.environ.get('TMPDIR', '/tmp/ekasteleyn')) / "aurora_hres_latents" / filename]
            file_path = None
            for p in possible_paths:
                if p.exists():
                    file_path = p
                    break
            if file_path is None and s3_client is not None:
                s3_key = f"aurora_hres_validation/{filename}"
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        s3_client.download_file("ekasteleyn-aurora-predictions", s3_key, tmp.name)
                        file_path = Path(tmp.name)
                except: pass
            if file_path is None or not file_path.exists(): continue
            try: tensors.append(torch.load(file_path, weights_only=True, map_location='cpu').float())
            except: pass
        if not tensors: return None
        stacked = torch.stack(tensors, dim=0)
        mean_val = torch.nanmean(stacked, dim=0)
        return torch.nan_to_num(mean_val, nan=0.0, posinf=0.0, neginf=0.0)

    mean_active = load_mean_latent(active_dates)
    mean_neutral = load_mean_latent(neutral_dates)
    
    if mean_active is None or mean_neutral is None:
        print("ERROR: No latent files found for Active or Neutral dates. Cannot compute steering vector.")
        print("  Make sure extract_latents_hres.py has been run first and latents are on S3 or in TMPDIR.")
        sys.exit(1)
    else:
        delta_v = mean_active - mean_neutral
        delta_v = torch.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"delta_v shape: {delta_v.shape}, ndim={delta_v.ndim}")

    # Apply Latent Z-Index Filter
    # Shape from encoder_layers[2] hook: [1, seq_len, C] (3D)
    # Shape from 3D Swin backbone:       [1, Z, H, W, C] (5D)
    ablated_delta_v = torch.zeros_like(delta_v)
    if delta_v.ndim == 5:
        # 3D Swin shape: [1, Z, H, W, C] — steer one Z level only
        if args.latent_idx >= delta_v.shape[1]:
            print(f"ERROR: --latent-idx {args.latent_idx} out of range for Z dim size {delta_v.shape[1]}")
            sys.exit(1)
        ablated_delta_v[:, args.latent_idx, :, :, :] = delta_v[:, args.latent_idx, :, :, :]
        print(f"Applied Z-level ablation: keeping level {args.latent_idx} of {delta_v.shape[1]}")
    elif delta_v.ndim == 3:
        # Encoder output shape: [1, seq_len, C] — partition channels into 3 equal slices
        C = delta_v.shape[2]
        slice_size = C // 3
        c_start = args.latent_idx * slice_size
        c_end = c_start + slice_size if args.latent_idx < 2 else C  # last slice gets remainder
        if c_start >= C:
            print(f"ERROR: --latent-idx {args.latent_idx} out of range for C={C}")
            sys.exit(1)
        ablated_delta_v[:, :, c_start:c_end] = delta_v[:, :, c_start:c_end]
        print(f"3D latent: steering channel slice [{c_start}:{c_end}] of {C} (latent-idx={args.latent_idx})")
    else:
        print(f"ERROR: Unexpected delta_v ndim={delta_v.ndim}, shape={delta_v.shape}. Cannot apply latent-idx ablation.")
        sys.exit(1)

    masked_delta_v = apply_spatial_mask_to_delta(ablated_delta_v, mask_region=args.mask_region, polar_lat_min=args.polar_lat_min, hemisphere=args.hemisphere)
    steering_vec = masked_delta_v

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vec_name = f"{args.phenomenon}_1encoder(2)_la{args.latent_idx}"
    steering_vec_path = output_dir / f"{vec_name}.pt"
    torch.save(steering_vec.cpu(), steering_vec_path)

    def make_intervention_hook(steering_vec, alpha=1.0):
        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            s_vec = steering_vec.to(dtype=x.dtype, device=x.device)
            new_x = x + (alpha * s_vec)
            if is_tuple: return (new_x,) + output[1:]
            return new_x
        return hook
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data_dir:
        download_dir = Path(args.data_dir)
    else:
        shared_scratch = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
        download_dir = shared_scratch if shared_scratch.parent.exists() else Path(os.environ.get("TMPDIR", "/tmp")) / "aurora_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using data dir: {download_dir}")
    print("Downloading static data (if needed)...")
    download_static(download_dir)

    base_date = neutral_dates.iloc[0] if not args.base_date else args.base_date
    base_day_str = base_date if args.base_date else f"{int(base_date['Year']):04d}-{int(base_date['Month']):02d}-{int(base_date['Day']):02d}"
    print(f"Base date: {base_day_str}")

    print("Downloading HRES data (if needed)...")
    download_data(base_day_str, download_dir)
    if args.init_hour == 0:
        download_data((pd.to_datetime(base_day_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), download_dir)

    print("Preparing batch...")
    batch = prepare_batch(base_day_str, download_dir, init_hour=args.init_hour)
    date_tag = base_day_str.replace("-", "")
    init_tag = f"{args.init_hour:02d}00"
    print(f"Batch ready. date_tag={date_tag}, init_tag={init_tag}")

    print(f"Loading Aurora model on {device}...")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)
    print("Model loaded.")

    if isinstance(batch, tuple):
        batch = tuple(t.to(device) if hasattr(t, 'to') else t for t in batch)
    else: batch = batch.to(device)

    print(f"Running inference for {len(args.alphas)} alphas: {args.alphas}")
    for alpha_val in args.alphas:
        print(f"  Applying hook with alpha={alpha_val}...")
        hook_handle = model.backbone.encoder_layers[2].register_forward_hook(make_intervention_hook(steering_vec, alpha=alpha_val))
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=args.steps):
                pred_batch = pred

        pred_batch = pred_batch.to("cpu")
        hook_handle.remove()

        ds = batch_to_dataset(pred_batch, step=args.steps)
        lat_tag = str(args.polar_lat_min).replace(".", "p")
        mask_tag = "nomask" if args.mask_region == "none" else f"polar_{args.hemisphere}_lat{lat_tag}"
        output_filename = output_dir / f"steered_{vec_name}{suffix_str}_{date_tag}_{init_tag}_{mask_tag}_alpha_{alpha_val}.nc"
        ds.to_netcdf(output_filename)

if __name__ == "__main__":
    main()
