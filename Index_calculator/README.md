# Oscillation Calculator

This folder contains a clean set of scripts for calculating the empirical orthogonal functions (EOFs) and evaluating atmospheric and oceanic indices from NetCDF datasets. The supported indices are:
- **NAO** (North Atlantic Oscillation)
- **PNA** (Pacific-North American Pattern)
- **AAO** (Antarctic Oscillation)
- **AO** (Arctic Oscillation)
- **ENSO / ONI** (El Niño-Southern Oscillation / Oceanic Niño Index)
- **MJO / RMM** (Madden-Julian Oscillation / Real-time Multivariate MJO index)

## Prerequisites

Make sure you have the following packages installed in your Python environment:
```bash
pip install xarray numpy pandas eofs windspharm
```
*(Note: `windspharm` is specifically required for MJO calculations to compute velocity potential.)*

## 1. EOF Generation

Before you can calculate indices for the atmospheric teleconnections (NAO, PNA, AAO, AO) or MJO, you need to extract the static loading patterns (EOFs) from a climatology dataset (like ERA5).

### Standard Atmospheric Modes (NAO, PNA, AAO, AO)
Generate the loading patterns for the standard atmospheric modes:
```bash
python generate_eofs.py --input <path_to_era5_climatology.zarr> --output_dir ./indices/
```
You can optionally specify `--index NAO` to generate just one specific mode.

### Madden-Julian Oscillation (MJO)
Generate the combined EOFs (RMM1 and RMM2) from U850, U200, and V200 winds. This script utilizes multiprocessing.
```bash
python generate_mjo_eof.py --input <path_to_era5_climatology.zarr> --output_dir ./indices/
```

## 2. Index Calculation

Once the static EOF patterns are generated, you can evaluate the index values for any target forecast or state file (e.g., from a machine learning model output) by comparing it against the climatology.

### Standard Atmospheric Indices (NAO, PNA, AAO, AO)
Calculate the indices for a specific target NetCDF file:
```bash
python calculate_indices.py \
    --target <path_to_target_forecast.nc> \
    --climatology <path_to_climatology.nc_or_zarr> \
    --eofs_dir ./indices/
```

### ENSO (Oceanic Niño Index)
ENSO doesn't require an EOF. It calculates the area-weighted SST anomalies directly:
```bash
python calculate_enso.py \
    --target <path_to_target_forecast.nc> \
    --climatology <path_to_climatology.nc_or_zarr>
```

### MJO (RMM Phase and Amplitude)
Calculate the RMM phase and amplitude using the pre-calculated MJO EOFs:
```bash
python calculate_mjo.py \
    --target <path_to_target_forecast.nc> \
    --climatology <path_to_climatology.nc_or_zarr> \
    --eof_file ./indices/mjo_loading_pattern.nc
```
