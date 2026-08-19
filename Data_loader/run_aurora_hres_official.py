#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Aurora 0.25° Fine-Tuned with IFS HRES T0 - Following Official Microsoft Example

This script follows the official Microsoft Aurora example exactly:
https://microsoft.github.io/aurora/example_hres_t0.html

Runs Aurora on 3 dates in 2022 and compares with WeatherBench2.
"""

from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import torch
# Force strict FP32 math and determinism
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import xarray as xr
from huggingface_hub import hf_hub_download
import pickle

from aurora import Aurora, Batch, Metadata, rollout

# ============================================================================
# Configuration
# ============================================================================

DOWNLOAD_PATH = Path.home() / "downloads" / "hres_t0"
OUTPUT_DIR = Path(f"/scratch-shared/{os.environ.get('USER', 'ekasteleyn')}/aurora_hres_validation")

# WeatherBench2 HRES T0 data
WB2_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

# Dates to process
DATES = ["2022-01-15", "2022-05-15", "2022-09-15"]

# ============================================================================
# Download Data (following official example)
# ============================================================================

def download_data(day: str, download_path: Path):
    """Download HRES T0 data for a specific day."""
    download_path.mkdir(parents=True, exist_ok=True)
    
    ds = xr.open_zarr(fsspec.get_mapper(WB2_URL), chunks=None)
    
    # Download surface-level variables
    if not (download_path / f"{day}-surface-level.nc").exists():
        print(f"  Downloading surface variables for {day}...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(download_path / f"{day}-surface-level.nc"))
        print(f"    ✓ Surface-level variables downloaded")
    
    # Download atmospheric variables
    if not (download_path / f"{day}-atmospheric.nc").exists():
        print(f"  Downloading atmospheric variables for {day}...")
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(download_path / f"{day}-atmospheric.nc"))
        print(f"    ✓ Atmospheric variables downloaded")


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
        print(f"    ✓ Static variables downloaded")


# ============================================================================
# Prepare Batch (following official example EXACTLY)
# ============================================================================

def prepare_batch(day: str, download_path: Path) -> Batch:
    """Prepare batch following official Microsoft example."""
    
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")
    
    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare a variable.
        
        This does the following things:
        * Select time steps at 06:00 and 12:00 (indices 1,2) so init=12:00 UTC.
        * This matches WB2 Aurora which has init times at 00:00 and 12:00.
        * Insert an empty batch dimension with `[None]`.
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Copy the data, because the data must be contiguous when converting to PyTorch.
        * Convert to PyTorch.
        """
        return torch.from_numpy(x[1:3][None][..., ::-1, :].copy())
    
    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            # They don't need to be flipped along the latitude dimension, because they
            # are from ERA5.
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
            # Flip the latitudes! We need to copy because converting to PyTorch,
            # because the data must be contiguous.
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element. Select element 2, corresponding to
            # time 12:00 (init time for WB2 comparison).
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[2],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )
    
    return batch


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


# ============================================================================
# Comparison with WB2
# ============================================================================

def compute_rmse(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> float:
    """Compute latitude-weighted RMSE."""
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    diff_sq = (pred - truth) ** 2
    if diff_sq.ndim == 2:
        weighted = diff_sq * weights[:, None]
    else:
        weighted = diff_sq
    return float(np.sqrt(np.nanmean(weighted)))


def compare_with_wb2(predictions: dict, dates: list) -> pd.DataFrame:
    """Compare predictions with WB2 Aurora benchmark."""
    print("\n" + "=" * 70)
    print("  COMPARING WITH WEATHERBENCH2 AURORA")
    print("=" * 70)
    
    wb2 = xr.open_zarr(fsspec.get_mapper("gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"), chunks=None)
    print(f"  WB2 Aurora levels: {list(wb2.level.values)}")
    
    results = []
    
    for date_str in dates:
        # Init time is 12:00 UTC (we use times 06:00 and 12:00, init is 12:00)
        # This matches WB2 Aurora init times at 00:00 and 12:00
        init_time = pd.to_datetime(f"{date_str}T12:00:00")
        
        for step in [1, 2, 3]:
            key = f"{date_str}_step{step}"
            if key not in predictions:
                continue
            
            pred_ds = predictions[key]
            lead_h = step * 6
            lead_td = np.timedelta64(lead_h, 'h')
            
            try:
                wb2_slice = wb2.sel(time=init_time, prediction_timedelta=lead_td)
                lat = pred_ds.latitude.values
                
                # Check if latitude directions mismatch
                wb2_lat = wb2_slice.latitude.values
                pred_is_descending = lat[0] > lat[-1]
                wb2_is_descending = wb2_lat[0] > wb2_lat[-1]
                need_flip = pred_is_descending != wb2_is_descending

                for level in [500, 700, 850]:
                    # Geopotential
                    pred_z = pred_ds['z'].sel(level=level).values
                    wb2_z = wb2_slice['geopotential\t'].sel(level=level).values
                    
                    # Temperature
                    pred_t = pred_ds['t'].sel(level=level).values
                    wb2_t = wb2_slice['temperature'].sel(level=level).values

                    # Flip WB2 if latitude directions mismatch
                    if need_flip:
                        wb2_z = wb2_z[::-1, :]
                        wb2_t = wb2_t[::-1, :]

                    rmse_z = compute_rmse(pred_z, wb2_z, lat)
                    rmse_t = compute_rmse(pred_t, wb2_t, lat)
                    
                    results.append({
                        'date': date_str,
                        'step': step,
                        'lead_hours': lead_h,
                        'level': level,
                        'z_rmse': rmse_z,
                        't_rmse': rmse_t,
                    })
                
                print(f"  ✓ {date_str} step {step} (+{lead_h}h)")
                
            except Exception as e:
                print(f"  ⚠ {date_str} step {step}: {e}")
    
    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", default=DATES)
    parser.add_argument("--num-steps", type=int, default=3)
    args = parser.parse_args()
    
    download_path = DOWNLOAD_PATH
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  AURORA 0.25° FINE-TUNED - IFS HRES T0")
    print("  Following official Microsoft example")
    print("=" * 70)
    print(f"  Dates: {args.dates}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Output: {output_dir}")
    print("=" * 70)
    
    # Download static variables
    print("\n[1/4] Downloading static variables...")
    download_static(download_path)
    
    # Download data for all dates
    print("\n[2/4] Downloading HRES T0 data...")
    for day in args.dates:
        download_data(day, download_path)
    
    # Load model (following official example)
    print("\n[3/4] Loading model...")
    from aurora import Aurora, rollout
    
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to("cuda")
    print("  ✓ Model loaded")
    
    # Run predictions
    print("\n[4/4] Running predictions...")
    all_predictions = {}
    
    for day in args.dates:
        print(f"\n  Processing {day}...")
        
        # Prepare batch (following official example)
        batch = prepare_batch(day, download_path)
        print(f"    Init time: {batch.metadata.time[0]}")
        print(f"    Levels: {batch.metadata.atmos_levels}")
        
        # Run rollout (following official example EXACTLY)
        with torch.inference_mode():
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=args.num_steps)]
        
        # Save predictions
        for step, pred in enumerate(preds, 1):
            pred_ds = batch_to_dataset(pred, step)
            
            key = f"{day}_step{step}"
            all_predictions[key] = pred_ds
            
            out_file = output_dir / f"aurora_hres_{day.replace('-', '')}_step{step:02d}.nc"
            pred_ds.to_netcdf(out_file)
            print(f"    ✓ Step {step} (+{step*6}h): {out_file.name}")
        
        # Clear memory
        del preds, batch
        torch.cuda.empty_cache()
    
    # Move model to CPU to free GPU memory
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    # Compare with WB2
    print("\nComparing with WeatherBench2...")
    df = compare_with_wb2(all_predictions, args.dates)
    
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("  RESULTS (RMSE vs WB2 Aurora)")
        print("=" * 70)
        print(df.to_string(index=False))
        
        print("\nMean RMSE by step:")
        print(df.groupby('step')[['z_rmse', 't_rmse']].mean())
        
        csv_path = output_dir / "wb2_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved: {csv_path}")
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
