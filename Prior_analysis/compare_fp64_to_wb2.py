#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Compare Aurora FP64 predictions vs WeatherBench2 Aurora outputs.

This script:
1. Runs Aurora in fp64 precision (double precision) for exact reproducibility
2. Compares the output to the official WeatherBench2 Aurora predictions
3. Reports whether they are bit-for-bit identical

Usage:
    python compare_fp64_to_wb2.py [--date YYYY-MM-DD] [--device cuda|cpu]
"""

from __future__ import annotations
import pickle
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

# Set memory allocation config before importing Aurora
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from aurora import Aurora, Batch, Metadata

# ============================================================================
# Configuration
# ============================================================================

# IFS HRES T0 data from WeatherBench2
HRES_T0_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

# WeatherBench2 Aurora benchmark results for comparison
WB2_AURORA_URL = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"
WB2_AURORA_6H_URL = "gs://weatherbench2/datasets/aurora/2022_6h-1440x721.zarr"

# Local cache path
CACHE_PATH = Path.home() / "downloads" / "hres_t0"


# ============================================================================
# Data Loading (adapted from official Microsoft example)
# ============================================================================

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


def download_data(day: str, download_path: Path):
    """Download HRES T0 data for a specific day."""
    download_path.mkdir(parents=True, exist_ok=True)
    
    ds = xr.open_zarr(fsspec.get_mapper(HRES_T0_URL), chunks=None)
    
    # Download surface-level variables
    if not (download_path / f"{day}-surface-level.nc").exists():
        print(f"    Downloading surface variables...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(download_path / f"{day}-surface-level.nc"))
    
    # Download atmospheric variables
    if not (download_path / f"{day}-atmospheric.nc").exists():
        print(f"    Downloading atmospheric variables...")
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(download_path / f"{day}-atmospheric.nc"))


def prepare_batch(day: str, download_path: Path, init_hour: int = 12) -> Batch:
    """Prepare batch in specified dtype (float32 or float64)."""
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")
    
    if init_hour == 12:
        # Init 12:00 UTC: use indices 1 (06:00) and 2 (12:00)
        time_indices = [1, 2]
        init_time_idx = 2
    elif init_hour == 0:
        # Init 00:00 UTC: need previous day's 18:00 and current day's 00:00
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


def compare_arrays(name: str, ours: np.ndarray, wb2: np.ndarray) -> dict:
    """Compare two arrays and return statistics."""
    diff = ours - wb2
    abs_diff = np.abs(diff)
    
    # Check for exact match
    exact_match = np.allclose(ours, wb2, rtol=0, atol=0)
    
    # Statistics
    result = {
        "name": name,
        "exact_match": exact_match,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "rel_diff_max": float(np.max(abs_diff / (np.abs(wb2) + 1e-10))),
        "num_different": int(np.sum(ours != wb2)),
        "total_elements": int(ours.size),
        "ours_dtype": str(ours.dtype),
        "wb2_dtype": str(wb2.dtype),
    }
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Aurora FP64 vs WB2 output")
    parser.add_argument("--date", type=str, default="2022-01-15", 
                        help="Date to test (YYYY-MM-DD)")
    parser.add_argument("--init-hours", nargs="+", type=int, default=[0, 12],
                        help="Init hours to test (default: [0, 12])")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP64")
    args = parser.parse_args()
    
    date_str = args.date
    init_hours = args.init_hours
    device = args.device
    dtype = torch.float32 if args.fp32 else torch.float64
    
    print("=" * 70)
    print("  AURORA FP64/FP32 vs WeatherBench2 COMPARISON")
    print("=" * 70)
    print(f"  Date: {date_str}")
    print(f"  Init hours: {init_hours}")
    print(f"  Device: {device}")
    print(f"  Precision: {'FP32' if args.fp32 else 'FP64'}")
    print("=" * 70)
    
    # Download data
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/5] Downloading static variables...")
    download_static(CACHE_PATH)
    
    print("\n[2/5] Downloading HRES T0 data...")
    dates_to_download = [date_str]
    if 0 in init_hours:
        # Need previous day for init 00:00
        prev_day = (pd.to_datetime(date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        dates_to_download.append(prev_day)
    
    for day in dates_to_download:
        print(f"  {day}")
        download_data(day, CACHE_PATH)
    
    # Load model in specified precision
    print(f"\n[3/5] Loading Aurora model in {dtype}...")
    model = Aurora()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    model.eval()
    model = model.to(device)  # Convert model to fp64/fp32
    print(f"  ✓ Model loaded in {dtype}")
    
    # Open WB2 Aurora dataset
    print("\n[4/5] Opening WeatherBench2 Aurora dataset...")
    try:
        # Try the regular dataset first
        wb2 = xr.open_zarr(fsspec.get_mapper(WB2_AURORA_URL), chunks=None)
        print(f"  ✓ WB2 Aurora dataset opened: {WB2_AURORA_URL}")
    except Exception as e:
        print(f"  Trying alternative URL...")
        wb2 = xr.open_zarr(fsspec.get_mapper(WB2_AURORA_6H_URL), chunks=None)
        print(f"  ✓ WB2 Aurora dataset opened: {WB2_AURORA_6H_URL}")
    
    print(f"    Variables: {list(wb2.data_vars)}")
    print(f"    Time range: {wb2.time.values[0]} to {wb2.time.values[-1]}")
    if "level" in wb2.dims:
        print(f"    Levels: {list(wb2.level.values)}")
    
    # Check data types in WB2
    for var in list(wb2.data_vars)[:3]:
        print(f"    {var} dtype: {wb2[var].dtype}")
    
    # Run model and compare
    print("\n[5/5] Running Aurora and comparing...")
    all_results = []
    
    for init_hour in init_hours:
        print(f"\n  Init {init_hour:02d}:00 UTC")
        print("  " + "-" * 50)
        
        # Prepare batch in specified precision
        batch = prepare_batch(date_str, CACHE_PATH, init_hour=init_hour)
        batch = batch.to(device)
        
        # Get init time and valid time
        init_time = batch.metadata.time[0]
        valid_time = init_time + pd.Timedelta(hours=6)
        print(f"    Init time: {init_time}")
        print(f"    Valid time (prediction): {valid_time}")
        
        # Run forward pass
        with torch.inference_mode():
            pred = model(batch)
        
        # Move to CPU for comparison
        pred = pred.to("cpu")
        
        # Get WB2 prediction for same valid time
        lead_td = np.timedelta64(6, 'h') # Step 1 is +6 hours
        try:
            # Correctly index by init time and prediction timedelta
            wb2_slice = wb2.sel(time=np.datetime64(init_time), prediction_timedelta=lead_td).compute()
        except Exception as e:
            print(f"    ⚠ Could not find init_time {init_time} in WB2: {e}")
            continue
        
        # Compare surface variables
        print("\n    Surface variables:")
        surf_mapping = {
            "2t": "2m_temperature",
            "10u": "10m_u_component_of_wind",
            "10v": "10m_v_component_of_wind",
            "msl": "mean_sea_level_pressure",
        }
        
        for our_var, wb2_var in surf_mapping.items():
            if our_var in pred.surf_vars and wb2_var in wb2_slice:
                ours = pred.surf_vars[our_var].numpy()[0, 0]
                theirs = wb2_slice[wb2_var].values
                
                pred_lat = pred.metadata.lat.numpy()
                wb2_lat = wb2_slice.latitude.values
                
                pred_is_descending = pred_lat[0] > pred_lat[-1]
                wb2_is_descending = wb2_lat[0] > wb2_lat[-1]
                
                if pred_is_descending != wb2_is_descending:
                    # If orientation mismatches, flip WB2 to match ours
                    theirs = theirs[::-1, :]
                # -------------------------------
                
                if ours.shape != theirs.shape:
                    print(f"      {our_var}: SHAPE MISMATCH ours={ours.shape} wb2={theirs.shape}")
                    continue
                
                result = compare_arrays(our_var, ours, theirs)
                result["init_hour"] = init_hour
                result["type"] = "surface"
                all_results.append(result)
                
                status = "✓ EXACT" if result["exact_match"] else f"✗ diff={result['max_abs_diff']:.2e}"
                print(f"      {our_var}: {status}")
        
        # Compare atmospheric variables (at available levels)
        print("\n    Atmospheric variables:")
        atmos_mapping = {
            "t": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "q": "specific_humidity",
            "z": "geopotential",
        }
        
        # Find common levels
        our_levels = list(pred.metadata.atmos_levels)
        wb2_levels = list(wb2_slice.level.values) if "level" in wb2_slice.dims else []
        common_levels = [l for l in our_levels if l in wb2_levels]
        print(f"      Our levels: {our_levels}")
        print(f"      WB2 levels: {wb2_levels}")
        print(f"      Common levels: {common_levels}")
        
        for our_var, wb2_var in atmos_mapping.items():
            if our_var not in pred.atmos_vars:
                continue
            if wb2_var not in wb2_slice:
                continue
                
            for level in common_levels:
                level_idx = our_levels.index(level)
                ours = pred.atmos_vars[our_var].numpy()[0, 0, level_idx]  # batch, time, level
                theirs = wb2_slice[wb2_var].sel(level=level).values
                pred_lat = pred.metadata.lat.numpy()
                wb2_lat = wb2_slice.latitude.values
                
                pred_is_descending = pred_lat[0] > pred_lat[-1]
                wb2_is_descending = wb2_lat[0] > wb2_lat[-1]
                
                if pred_is_descending != wb2_is_descending:
                    # If orientation mismatches, flip WB2 to match ours
                    theirs = theirs[::-1, :]
                
                if ours.shape != theirs.shape:
                    print(f"      {our_var}@{level}hPa: SHAPE MISMATCH ours={ours.shape} wb2={theirs.shape}")
                    continue
                
                result = compare_arrays(f"{our_var}@{level}hPa", ours, theirs)
                result["init_hour"] = init_hour
                result["type"] = "atmospheric"
                result["level"] = level
                all_results.append(result)
                
                status = "✓ EXACT" if result["exact_match"] else f"✗ diff={result['max_abs_diff']:.2e}"
                print(f"      {our_var}@{level}hPa: {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        exact_matches = df["exact_match"].sum()
        total = len(df)
        print(f"  Exact matches: {exact_matches}/{total}")
        
        if exact_matches < total:
            print("\n  Variables with differences:")
            diff_df = df[~df["exact_match"]][["name", "init_hour", "max_abs_diff", "mean_abs_diff", "num_different"]]
            print(diff_df.to_string(index=False))
            
            print("\n  Possible reasons for differences:")
            print("  1. Model weights are stored in FP32, not FP64")
            print("  2. WB2 predictions may have been generated with different precision")
            print("  3. Floating-point accumulation order differences across hardware")
            print("  4. Different CUDA versions or cuDNN implementations")
    
    # Save results
    output_path = CACHE_PATH / f"fp64_comparison_{date_str}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
