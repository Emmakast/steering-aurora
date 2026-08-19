#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Extract Aurora Latents from IFS HRES T0 Data

This script combines:
1. HRES T0 data loading from the official Microsoft Aurora example
2. Latent/attention extraction hooks from extract_latents.py

Uses Aurora 0.25° Fine-Tuned model with IFS HRES T0 initialization.
Extracts latent representations and attention weights from specified layers.
"""

from __future__ import annotations

import gc
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import torch
# Force strict FP32 math and determinism (must be set before any CUDA ops)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch.nn.functional as F
import xarray as xr
from huggingface_hub import hf_hub_download
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

from aurora import Aurora, Batch, Metadata, rollout

# ══════════════════════════════════════════════════════════════════════════════
# Custom SDPA for Attention Extraction
# ══════════════════════════════════════════════════════════════════════════════
original_sdpa = F.scaled_dot_product_attention
attention_activations = {}
current_capture_key = None


def custom_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    global current_capture_key, attention_activations
    res = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    
    if current_capture_key is not None:
        scale_factor = scale if scale is not None else (1.0 / (q.size(-1) ** 0.5))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(~attn_mask, float('-inf'))
            else:
                attn = attn + attn_mask
        attn_weights = torch.nn.functional.softmax(attn, dim=-1)
        # Cast to FP16 on GPU (fast), then move half the data across PCIe
        attention_activations[current_capture_key] = [attn_weights.detach().half().cpu()]
        
    return res


if not hasattr(F, '_patched'):
    F.scaled_dot_product_attention = custom_sdpa
    F._patched = True


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DOWNLOAD_PATH = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")

# Use SLURM's $TMPDIR (e.g. /scratch-node/ekasteleyn.1234567). 
# Fallback to local /tmp if running outside SLURM.
base_tmp = os.environ.get('TMPDIR', f"/tmp/{os.environ.get('USER', 'ekasteleyn')}")
OUTPUT_DIR = Path(base_tmp) / "aurora_hres_latents"

# WeatherBench2 HRES T0 data
WB2_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

# Target layers for latent extraction
TARGET_LAYERS: list[tuple[str, int]] = [
    ("perceiver", 0),  
    ("encoder", 0),
    ("encoder", 1),
    ("encoder", 2),
]

# Default dates (can be overridden via command line or CSV)
DEFAULT_DATES = ["2022-01-15", "2022-05-15", "2022-09-15"]


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading (from official Microsoft example)
# ══════════════════════════════════════════════════════════════════════════════

def download_data(day: str, download_path: Path):
    """Download HRES T0 data for a specific day."""
    download_path.mkdir(parents=True, exist_ok=True)

    surf_path = download_path / f"{day}-surface-level.nc"
    atmos_path = download_path / f"{day}-atmospheric.nc"

    # Short-circuit: if both files already exist, skip GCS entirely.
    # Opening the zarr on GCS is slow even when no download is needed.
    if surf_path.exists() and atmos_path.exists():
        return

    # Only open the remote zarr store if at least one file is missing.
    ds = xr.open_zarr(fsspec.get_mapper(WB2_URL), chunks=None)

    # Download surface-level variables
    if not surf_path.exists():
        print(f"    Downloading surface variables...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(surf_path))

    # Download atmospheric variables
    if not atmos_path.exists():
        print(f"    Downloading atmospheric variables...")
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(atmos_path))


def download_static(download_path: Path):
    """Download static variables from HuggingFace."""
    if not (download_path / "static.nc").exists():
        print("  Downloading static variables from HuggingFace...")
        path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
        with open(path, "rb") as f:
            static_vars = pickle.load(f)
        
        ds_static = xr.Dataset(
            data_vars={k: (["latitude", "longitude"], v) for k, v in static_vars.items()},
            coords={
                "latitude": ("latitude", np.linspace(90, -90, 721)),
                "longitude": ("longitude", np.linspace(0, 360, 1440, endpoint=False)),
            },
        )
        ds_static.to_netcdf(str(download_path / "static.nc"))
        print("    ✓ Static variables cached")


def prepare_batch(day: str, download_path: Path, init_hour: int = 12, static_vars_ds: xr.Dataset = None) -> Batch:
    """Prepare batch following official Microsoft example.
    
    Args:
        day: Date string (YYYY-MM-DD)
        download_path: Path to downloaded data
        init_hour: Initialization hour (0 or 12 UTC)
            - init_hour=12: uses times 06:00 and 12:00 (indices 1,2)
            - init_hour=0: uses times 18:00 (prev day) and 00:00 (indices 3,0 effectively)
              For simplicity, we use indices 0,1 which is 00:00 and 06:00, init=06:00
              Actually for init=00:00, we need prev day 18:00 + current day 00:00.
        static_vars_ds: Pre-loaded static variables dataset (to avoid repeated file access)
              
    Note: For init_hour=0, this requires the previous day's data to be available.
    The HRES T0 data has times: 00:00, 06:00, 12:00, 18:00 (indices 0,1,2,3).
    - init=12:00 uses t-6h=06:00 (idx 1) and t=12:00 (idx 2)
    - init=00:00 uses t-6h=18:00 (prev day idx 3) and t=00:00 (idx 0)
    """
    if static_vars_ds is None:
        static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")
    
    if init_hour == 12:
        # Init 12:00 UTC: use indices 1 (06:00) and 2 (12:00)
        time_indices = [1, 2]
        init_time_idx = 2
    elif init_hour == 0:
        # Init 00:00 UTC: need previous day's 18:00 and current day's 00:00
        # Load previous day data
        prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prev_surf_ds = xr.open_dataset(download_path / f"{prev_day}-surface-level.nc", engine="netcdf4")
        prev_atmos_ds = xr.open_dataset(download_path / f"{prev_day}-atmospheric.nc", engine="netcdf4")
        
        def _prepare_init00(x_prev: np.ndarray, x_curr: np.ndarray) -> torch.Tensor:
            """Prepare for init=00:00: prev day 18:00 (idx 3) + current day 00:00 (idx 0)."""
            combined = np.stack([x_prev[3], x_curr[0]], axis=0)
            return torch.from_numpy(combined[None][..., ::-1, :].copy())
        
        batch = Batch(
            surf_vars={
                "2t": _prepare_init00(prev_surf_ds["2m_temperature"].values, surf_vars_ds["2m_temperature"].values),
                "10u": _prepare_init00(prev_surf_ds["10m_u_component_of_wind"].values, surf_vars_ds["10m_u_component_of_wind"].values),
                "10v": _prepare_init00(prev_surf_ds["10m_v_component_of_wind"].values, surf_vars_ds["10m_v_component_of_wind"].values),
                "msl": _prepare_init00(prev_surf_ds["mean_sea_level_pressure"].values, surf_vars_ds["mean_sea_level_pressure"].values),
            },
            static_vars={
                "z": torch.from_numpy(static_vars_ds["z"].values),
                "slt": torch.from_numpy(static_vars_ds["slt"].values),
                "lsm": torch.from_numpy(static_vars_ds["lsm"].values),
            },
            atmos_vars={
                "t": _prepare_init00(prev_atmos_ds["temperature"].values, atmos_vars_ds["temperature"].values),
                "u": _prepare_init00(prev_atmos_ds["u_component_of_wind"].values, atmos_vars_ds["u_component_of_wind"].values),
                "v": _prepare_init00(prev_atmos_ds["v_component_of_wind"].values, atmos_vars_ds["v_component_of_wind"].values),
                "q": _prepare_init00(prev_atmos_ds["specific_humidity"].values, atmos_vars_ds["specific_humidity"].values),
                "z": _prepare_init00(prev_atmos_ds["geopotential"].values, atmos_vars_ds["geopotential"].values),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
                lon=torch.from_numpy(surf_vars_ds.longitude.values),
                # Init time is 00:00 of current day
                time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[0],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
            ),
        )
        prev_surf_ds.close()
        prev_atmos_ds.close()
        return batch
    else:
        raise ValueError(f"init_hour must be 0 or 12, got {init_hour}")
    
    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare a variable with specified time indices."""
        return torch.from_numpy(x[time_indices][None][..., ::-1, :].copy())
    
    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values),
            "slt": torch.from_numpy(static_vars_ds["slt"].values),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values),
        },
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[init_time_idx],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )
    
    return batch


# ══════════════════════════════════════════════════════════════════════════════
# Hook Registration (from extract_latents.py)
# ══════════════════════════════════════════════════════════════════════════════

def make_attn_pre_hook(key: str):
    def pre_hook(module, args):
        global current_capture_key
        current_capture_key = key
    return pre_hook


def make_attn_post_hook(key: str):
    def post_hook(module, args, output):
        global current_capture_key
        current_capture_key = None
    return post_hook


def unregister_hooks(handles: list):
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    handles.clear()


def register_hooks(
    model: torch.nn.Module,
    target_layers: list[tuple[str, int]],
) -> tuple[dict[str, torch.Tensor], list]:
    """Register forward hooks to capture latent activations and attention weights."""
    activations: dict[str, torch.Tensor] = {}
    handles: list = []

    for part, idx in target_layers:
        key = f"{part}_{idx}"

        def _make_hook(k: str):
            def _hook(_module, _input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                # Cast to FP16 on GPU (fast), then move half the data across PCIe
                activations[k] = tensor.detach().half().cpu()
            return _hook

        if part == "perceiver":
            module = model.encoder
            handles.append(module.register_forward_hook(_make_hook(key)))
            
            # Capture cross-attention between atmospheric levels and latent levels
            last_perceiver_attn = module.level_agg.layers[-1][0]
            attn_key = "perceiver_cross_attn"
            handles.append(last_perceiver_attn.register_forward_pre_hook(make_attn_pre_hook(attn_key)))
            handles.append(last_perceiver_attn.register_forward_hook(make_attn_post_hook(attn_key)))
            
        elif part == "encoder":
            module = model.backbone.encoder_layers[idx]
            handles.append(module.register_forward_hook(_make_hook(key)))
            
            # Capture attention weights from last block
            last_block_attn = module.blocks[-1].attn
            attn_key = f"encoder_{idx}_attn"
            handles.append(last_block_attn.register_forward_pre_hook(make_attn_pre_hook(attn_key)))
            handles.append(last_block_attn.register_forward_hook(make_attn_post_hook(attn_key)))
        else:
            raise ValueError(f"Unknown part '{part}'")

    return activations, handles
# ══════════════════════════════════════════════════════════════════════════════
# S3 Upload Helper
# ══════════════════════════════════════════════════════════════════════════════
def upload_to_s3(s3_client, local_path: Path, bucket_name: str, s3_folder: str):
    """Uploads a local file to S3 and optionally deletes the local copy."""
    if s3_client is None:
        return # Skip if S3 wasn't initialized
        
    s3_key = f"{s3_folder}/{local_path.name}"
    try:
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        print(f"      ↑ Uploaded to S3: s3://{bucket_name}/{s3_key}")
        
        # OPTIONAL: Delete local file to save scratch space! 
        # Uncomment the line below if you want to stream data without filling the disk
        # local_path.unlink() 
        
    except Exception as e:
        print(f"      ⚠ S3 Upload Failed for {local_path.name}: {e}")
# ============================================================================
# Convert prediction to xarray for saving
# ============================================================================

def batch_to_dataset(pred: Batch, step: int) -> xr.Dataset:
    """Convert Aurora Batch prediction to xarray Dataset."""
    lat = pred.metadata.lat.numpy()
    lon = pred.metadata.lon.numpy()
    levels = list(pred.metadata.atmos_levels)
    
    data_vars = {}
    
    # Surface variables
    for name, tensor in pred.surf_vars.items():
        arr = tensor.numpy()
        # Shape is (batch=1, time=1, lat, lon) -> (lat, lon)
        arr = arr[0, 0]
        data_vars[name] = (["latitude", "longitude"], arr)
    
    # Atmospheric variables
    for name, tensor in pred.atmos_vars.items():
        arr = tensor.numpy()
        # Shape is (batch=1, time=1, level, lat, lon) -> (level, lat, lon)
        arr = arr[0, 0]
        data_vars[name] = (["level", "latitude", "longitude"], arr)
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "latitude": lat,
            "longitude": lon,
            "level": levels,
        },
    )
    ds.attrs["valid_time"] = str(pred.metadata.time[0])
    ds.attrs["step"] = step
    ds.attrs["lead_hours"] = step * 6
    ds.attrs["model"] = "Aurora 0.25 Fine-Tuned"
    
    return ds

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Aurora latents from IFS HRES T0 data"
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Dates to process (YYYY-MM-DD). Default: use dates CSV or built-in defaults"
    )
    parser.add_argument(
        "--dates-csv", type=str, default=None,
        help="Path to CSV file with 'date' column"
    )
    parser.add_argument(
        "--num-steps", type=int, default=1,
        help="Number of rollout steps (default: 1 = single forward pass)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache directory for downloads (default: {DOWNLOAD_PATH})"
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Also save prediction outputs as NetCDF"
    )
    parser.add_argument(
        "--init-hours", nargs="+", type=int, default=[12],
        help="Init hours to process (0 and/or 12). Default: [12]"
    )
    parser.add_argument(
        "--latent-init-hour", type=int, default=None,
        help="Only extract latents for this init hour (default: same as first --init-hours value)"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile() for faster inference. WARNING: alters floating-point "
             "accumulation order, breaking bit-exact reproducibility with WB2 Aurora."
    )
    parser.add_argument(
        "--fp64", action="store_true",
        help="Run model and save predictions in float64 (for precision testing)"
    )
    args = parser.parse_args()
    
    # Default latent_init_hour to first init_hour if not specified
    if args.latent_init_hour is None:
        args.latent_init_hour = args.init_hours[0]
    
    # Determine dates
    if args.dates:
        dates = args.dates
    elif args.dates_csv:
        df = pd.read_csv(args.dates_csv)
        if "date" in df.columns:
            dates = df["date"].tolist()
        elif "Year" in df.columns and "Month" in df.columns and "Day" in df.columns:
            dates = [f"{int(r.Year):04d}-{int(r.Month):02d}-{int(r.Day):02d}" for _, r in df.iterrows()]
        else:
            raise ValueError("CSV must contain either 'date' or 'Year', 'Month', 'Day' columns.")
    else:
        dates = DEFAULT_DATES
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    download_path = Path(args.cache_dir) if args.cache_dir else DOWNLOAD_PATH
    
    output_dir.mkdir(parents=True, exist_ok=True)
    download_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_dotenv("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/.env")
    s3_client = None
    bucket_name = "ekasteleyn-aurora-predictions"
    s3_folder = "aurora_hres_validation" 

    if HAS_BOTO3 and os.getenv('UVA_S3_ACCESS_KEY') and os.getenv('UVA_S3_SECRET_KEY'):
        s3_client = boto3.client('s3',
            endpoint_url='https://ceph-gw.science.uva.nl:8000',
            aws_access_key_id=os.getenv('UVA_S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('UVA_S3_SECRET_KEY')
        )
        print("  ✓ S3 Client initialized securely via .env")
    else:
        print("  ⚠ S3 credentials not found in .env. Files will only be saved locally.")
    
    print("=" * 70)
    print("  AURORA LATENT EXTRACTION - IFS HRES T0")
    print("=" * 70)
    print(f"  Dates: {len(dates)} dates")
    print(f"  Init hours: {args.init_hours}")
    print(f"  Latent extraction init: {args.latent_init_hour}:00 UTC only")
    print(f"  Rollout steps: {args.num_steps}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Cache: {download_path}")
    print(f"  Target layers: {TARGET_LAYERS}")
    print("=" * 70)
    
    # Download static variables
    print("\n[1/4] Downloading static variables...")
    download_static(download_path)
    
    # Download data for all dates (and previous days if init_hour=0 is requested)
    print("\n[2/4] Downloading HRES T0 data...")
    dates_to_download = set(dates)
    if 0 in args.init_hours:
        # For init=00:00, we need the previous day's data
        for day in dates:
            prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            dates_to_download.add(prev_day)
    for day in sorted(dates_to_download):
        print(f"  {day}")
        download_data(day, download_path)
    
    # Load model
    print("\n[3/4] Loading Aurora 0.25° Fine-Tuned model...")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)
    
    # Optional: compile model for faster inference
    if args.compile:
        print("  ⚠ WARNING: torch.compile() alters FP accumulation order, breaking reproducibility!")
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("  ✓ Model compiled")
    
    print("  ✓ Model loaded")
    
    # Thread pool for async file saving
    save_executor = ThreadPoolExecutor(max_workers=4)
    pending_saves = []
    
    # def async_save_pt(tensor, path, s3_client, bucket, folder):
    #     """Save tensor and upload to S3 in background."""
    #     torch.save(tensor, path)
    #     if s3_client:
    #         upload_to_s3(s3_client, path, bucket, folder)
    
    # def async_save_nc(ds, path, s3_client, bucket, folder):
    #     """Save dataset and upload to S3 in background."""
    #     ds.to_netcdf(path)
    #     if s3_client:
    #         upload_to_s3(s3_client, path, bucket, folder)
    
    # Process each date and init hour
    total_runs = len(dates) * len(args.init_hours)
    print(f"\n[4/4] Processing {total_runs} runs ({len(dates)} dates × {len(args.init_hours)} init times)...")
    
    # Load static data once to avoid race conditions with background I/O
    print("  Loading static variables into memory...")
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
    print("  ✓ Static variables loaded")
    
    run_idx = 0
    total_start_time = time.time()
    for date_str in dates:
        for init_hour in args.init_hours:
            run_idx += 1
            run_start_time = time.time()
            extract_latents = (init_hour == args.latent_init_hour)
            
            # Progress info with timestamp and ETA
            elapsed_total = time.time() - total_start_time
            if run_idx > 1:
                avg_per_run = elapsed_total / (run_idx - 1)
                remaining_runs = total_runs - run_idx + 1
                eta_seconds = avg_per_run * remaining_runs
                eta_str = f" | ETA: {eta_seconds/60:.1f} min"
            else:
                eta_str = ""
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Check if it already exists on S3 to save time
            if extract_latents and s3_client is not None:
                date_fmt = date_str.replace("-", "")
                init_fmt = f"{init_hour:02d}00"
                s3_key = f"{s3_folder}/latent_{date_fmt}_{init_fmt}_encoder_2.pt"
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                    print(f"\n  [{timestamp}] [{run_idx}/{total_runs}] {date_str} init {init_hour:02d}:00 UTC [skipped - already on S3]")
                    continue
                except Exception:
                    pass

            print(f"\n  [{timestamp}] [{run_idx}/{total_runs}] {date_str} init {init_hour:02d}:00 UTC", end="")
            if extract_latents:
                print(f" [+latents]{eta_str}")
            else:
                print(f" [predictions only]{eta_str}")
            
            try:
                # Register hooks only if extracting latents for this init hour
                if extract_latents:
                    activations, handles = register_hooks(model, TARGET_LAYERS)
                else:
                    activations = {}
                    handles = []
                
                # Prepare batch
                batch = prepare_batch(date_str, download_path, init_hour=init_hour, static_vars_ds=static_vars_ds)
                init_time = batch.metadata.time[0]
                print(f"    Init time: {init_time}")
                
                batch = batch.to(device)
                
                # Format init time for filenames (e.g., "20220115_1200")
                init_dt = init_time
                date_fmt = init_dt.strftime("%Y%m%d")
                init_fmt = init_dt.strftime("%H%M")
                
                # Run rollout with hooks active for step 1
                # Using rollout() ensures time metadata advances correctly for solar forcings
                with torch.inference_mode():
                    step = 0
                    for pred in rollout(model, batch, steps=args.num_steps):
                        step += 1
                        pred_cpu = pred.to("cpu")
                        
                        # # Save prediction
                        # if args.save_predictions:
                        #     lead_hours = step * 6
                        #     pred_ds = batch_to_dataset(pred_cpu, step)
                        #     out_path = output_dir / f"aurora_pred_{date_fmt}_{init_fmt}_step{step:02d}_{lead_hours:03d}h.nc"
                        #     pending_saves.append(save_executor.submit(
                        #         async_save_nc, pred_ds, out_path, s3_client, bucket_name, s3_folder
                        #     ))
                        #     print(f"    pred step {step} (+{lead_hours}h) -> {out_path.name}")
                        #     del pred_ds
                        
                        # # Save latents only from step 1 (already FP16 from hooks)
                        # if step == 1 and extract_latents:
                        #     for key, tensor in activations.items():
                        #         out_path = output_dir / f"latent_{date_fmt}_{init_fmt}_{key}.pt"
                        #         pending_saves.append(save_executor.submit(
                        #             async_save_pt, tensor, out_path, s3_client, bucket_name, s3_folder
                        #         ))
                        #         print(f"    {key}: shape={tuple(tensor.shape)} -> {out_path.name}")
                            
                        #     for key, attn_list in attention_activations.items():
                        #         out_path = output_dir / f"attn_{date_fmt}_{init_fmt}_{key}.pt"
                        #         stacked_attn = torch.cat(attn_list, dim=0) if len(attn_list) > 1 else attn_list[0]
                        #         pending_saves.append(save_executor.submit(
                        #             async_save_pt, stacked_attn, out_path, s3_client, bucket_name, s3_folder
                        #         ))
                        #         print(f"    {key}: shape={tuple(stacked_attn.shape)} -> {out_path.name}")
                        # Save prediction
                        if args.save_predictions:
                            lead_hours = step * 6
                            pred_ds = batch_to_dataset(pred_cpu, step)
                            out_path = output_dir / f"aurora_pred_{date_fmt}_{init_fmt}_step{step:02d}_{lead_hours:03d}h.nc"
                            
                            # 1. Write to disk synchronously on the main thread (Thread-safe & Memory-safe)
                            pred_ds.to_netcdf(out_path)
                            print(f"    pred step {step} (+{lead_hours}h) -> {out_path.name} (Disk)")
                            
                            # 2. Upload to S3 asynchronously in the background
                            if s3_client:
                                pending_saves.append(save_executor.submit(
                                    upload_to_s3, s3_client, out_path, bucket_name, s3_folder
                                ))
                            
                            # Safe to delete now that disk write is finished
                            del pred_ds
                        
                        # Save latents only from step 1
                        if step == 1 and extract_latents:
                            for key, tensor in activations.items():
                                out_path = output_dir / f"latent_{date_fmt}_{init_fmt}_{key}.pt"
                                
                                # 1. Write to disk synchronously
                                torch.save(tensor, out_path)
                                print(f"    {key}: shape={tuple(tensor.shape)} -> {out_path.name} (Disk)")
                                
                                # 2. Upload asynchronously
                                if s3_client:
                                    pending_saves.append(save_executor.submit(
                                        upload_to_s3, s3_client, out_path, bucket_name, s3_folder
                                    ))
                            
                            for key, attn_list in attention_activations.items():
                                out_path = output_dir / f"attn_{date_fmt}_{init_fmt}_{key}.pt"
                                stacked_attn = torch.cat(attn_list, dim=0) if len(attn_list) > 1 else attn_list[0]
                                
                                # 1. Write to disk synchronously
                                torch.save(stacked_attn, out_path)
                                print(f"    {key}: shape={tuple(stacked_attn.shape)} -> {out_path.name} (Disk)")
                                
                                # 2. Upload asynchronously
                                if s3_client:
                                    pending_saves.append(save_executor.submit(
                                        upload_to_s3, s3_client, out_path, bucket_name, s3_folder
                                    ))
                            
                            # Disable hooks after step 1
                            unregister_hooks(handles)
                            activations.clear()
                            attention_activations.clear()
                        
                        del pred_cpu
                        del pred  # Also delete GPU tensor
                
                del batch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all GPU ops complete
                gc.collect()
                
                # Log timing for this run
                run_elapsed = time.time() - run_start_time
                print(f"    ✓ Done in {run_elapsed:.1f}s")
                
            except Exception as e:
                print(f"    ⚠ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Wait for all pending saves to complete
    print("\n  Waiting for background saves to complete...")
    for future in pending_saves:
        future.result()
    save_executor.shutdown()
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Output saved to: {output_dir}")
    pt_files = list(output_dir.glob('*.pt'))
    nc_files = list(output_dir.glob('*.nc'))
    print(f"  Files: {len(pt_files)} .pt files, {len(nc_files)} .nc files")


if __name__ == "__main__":
    main()
