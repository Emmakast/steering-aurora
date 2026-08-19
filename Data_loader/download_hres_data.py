#!/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
"""
Download HRES T0 data for Aurora latent extraction.

This script pre-downloads all required data so the GPU job can start
processing immediately without wasting GPU time on I/O.
"""

import argparse
import pickle
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import hf_hub_download

# Default paths
DOWNLOAD_PATH = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
WB2_HRES_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
WB2_ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"


def _is_valid_netcdf(path: Path) -> bool:
    """Lightweight validity check to avoid reusing truncated/corrupt files."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            _ = tuple(ds.dims.items())  # force metadata read
        return True
    except Exception:
        return False


def download_static(download_path: Path):
    """Download static variables from HuggingFace."""
    static_path = download_path / "static.nc"
    if _is_valid_netcdf(static_path):
        print("  ✓ Static variables already cached")
        return
    if static_path.exists():
        print("  ⚠ Found invalid static.nc, re-downloading...")
        static_path.unlink(missing_ok=True)

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
    tmp_path = static_path.with_suffix(".nc.tmp")
    ds_static.to_netcdf(str(tmp_path), engine="h5netcdf")
    tmp_path.replace(static_path)
    print("  ✓ Static variables downloaded")


def download_data(day: str, download_path: Path, ds: xr.Dataset, source: str = "HRES"):
    """Download HRES T0 or ERA5 data for a specific day."""
    surf_path = download_path / f"{day}-surface-level.nc"
    atmos_path = download_path / f"{day}-atmospheric.nc"

    # Validate cached files; remove invalid ones so they can be re-downloaded
    surf_ok = _is_valid_netcdf(surf_path)
    atmos_ok = _is_valid_netcdf(atmos_path)
    if surf_path.exists() and not surf_ok:
        print(f"  ⚠ Invalid cache file: {surf_path.name} (re-downloading)")
        surf_path.unlink(missing_ok=True)
    if atmos_path.exists() and not atmos_ok:
        print(f"  ⚠ Invalid cache file: {atmos_path.name} (re-downloading)")
        atmos_path.unlink(missing_ok=True)

    # Check if already downloaded
    if surf_ok and atmos_ok:
        print(f"  ✓ {day} already cached")
        return

    # Download surface-level variables
    if not surf_ok:
        print(f"  Downloading {day} surface variables ({source})...")
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        tmp_surf = surf_path.with_suffix(".nc.tmp")
        ds_surf.to_netcdf(str(tmp_surf), engine="h5netcdf")
        tmp_surf.replace(surf_path)

    # Download atmospheric variables
    if not atmos_ok:
        print(f"  Downloading {day} atmospheric variables ({source})...")
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        tmp_atmos = atmos_path.with_suffix(".nc.tmp")
        ds_atmos.to_netcdf(str(tmp_atmos), engine="h5netcdf")
        tmp_atmos.replace(atmos_path)

    print(f"  ✓ {day} downloaded")


def _dates_from_target_csv(
    csv_path: str,
    phenomenon: str | None = None,
    date_type: str | None = None,
) -> list[str]:
    """Load unique YYYY-MM-DD dates from target CSV with Year/Month/Day columns."""
    df = pd.read_csv(csv_path)

    # Support both title/lowercase variants
    y_col = "Year" if "Year" in df.columns else "year"
    m_col = "Month" if "Month" in df.columns else "month"
    d_col = "Day" if "Day" in df.columns else "day"

    if phenomenon and "Phenomenon" in df.columns:
        df = df[df["Phenomenon"] == phenomenon]

    if date_type and "Type" in df.columns:
        df = df[df["Type"] == date_type]

    dates = pd.to_datetime(
        df[[y_col, m_col, d_col]].rename(columns={y_col: "year", m_col: "month", d_col: "day"})
    ).dt.strftime("%Y-%m-%d")

    return sorted(dates.unique().tolist())


def main():
    parser = argparse.ArgumentParser(description="Download HRES T0 data for Aurora")
    parser.add_argument("--dates", nargs="+", default=None, help="Dates to download (YYYY-MM-DD)")
    parser.add_argument("--csv", type=str, default=None, help="Target dates CSV with Year/Month/Day")
    parser.add_argument("--csv-type", type=str, default=None, help="Optional Type filter for --csv (e.g., Active)")
    parser.add_argument("--neutral-csv", type=str, default=None, help="Optional second CSV to merge dates from")
    parser.add_argument("--neutral-type", type=str, default=None, help="Optional Type filter for --neutral-csv (e.g., Neutral)")
    parser.add_argument("--phenomenon", type=str, default=None, help="Optional phenomenon filter for CSVs (e.g., AO)")
    parser.add_argument("--cache-dir", type=str, default=None, help=f"Cache directory (default: {DOWNLOAD_PATH})")
    parser.add_argument("--include-prev-day", action="store_true",
                        help="Also download previous day for each date (needed for init_hour=0)")
    args = parser.parse_args()

    download_path = Path(args.cache_dir) if args.cache_dir else DOWNLOAD_PATH
    download_path.mkdir(parents=True, exist_ok=True)

    # Build date set from explicit --dates plus CSV inputs
    dates_seed = set(args.dates or [])
    if args.csv:
        dates_seed.update(_dates_from_target_csv(args.csv, args.phenomenon, args.csv_type))
    if args.neutral_csv:
        dates_seed.update(_dates_from_target_csv(args.neutral_csv, args.phenomenon, args.neutral_type))

    if not dates_seed:
        raise ValueError("No dates provided. Use --dates and/or --csv.")

    print("=" * 60)
    print("  HRES T0 DATA DOWNLOAD")
    print("=" * 60)
    print(f"  Seed dates: {len(dates_seed)}")
    print(f"  Cache: {download_path}")
    print("=" * 60)

    # Build full list of dates to download
    dates_to_download = set(dates_seed)
    if args.include_prev_day:
        for day in sorted(dates_seed):
            prev_day = (pd.to_datetime(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            dates_to_download.add(prev_day)

    dates_to_download = sorted(dates_to_download)
    print(f"\n[1/3] Total dates to download: {len(dates_to_download)}")
    
    # Download static variables
    print("\n[2/3] Static variables...")
    download_static(download_path)
    
    # Open WB2 datasets once
    print("\n[3/3] HRES T0 / ERA5 data...")
    print("  Opening WeatherBench2 datasets...")
    ds_hres = xr.open_zarr(fsspec.get_mapper(WB2_HRES_URL), chunks=None)
    ds_era5 = xr.open_zarr(fsspec.get_mapper(WB2_ERA5_URL), chunks=None)
    
    for day in dates_to_download:
        year = int(day.split("-")[0])
        if year < 2016:
            download_data(day, download_path, ds_era5, source="ERA5")
        else:
            download_data(day, download_path, ds_hres, source="HRES")
    
    ds_hres.close()
    ds_era5.close()
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Files cached at: {download_path}")
    
    # Show disk usage
    total_size = sum(f.stat().st_size for f in download_path.glob("*.nc"))
    print(f"  Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
