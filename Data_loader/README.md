# Aurora HRES Data Loader

This directory contains scripts for downloading High-Resolution (HRES) T0 data, running predictions using the Microsoft Aurora 0.25° Fine-Tuned model, and extracting latent representations and attention weights for steering experiments.

## Files

### 1. `download_hres_data.py`
A utility script to pre-download all required data from WeatherBench2 (GCS) and Hugging Face so that GPU jobs can start processing immediately without wasting GPU time on I/O.

**Features:**
- Downloads HRES T0 data (2016-2022) or ERA5 data (before 2016).
- Downloads static variables (topography, land-sea mask, etc.) from Hugging Face.
- Supports reading target dates from one or more CSV files, optionally filtering by phenomenon or type.
- Can optionally download the previous day's data (required for 00:00 UTC initialization).

**Usage:**
```bash
python download_hres_data.py --dates 2022-01-15 2022-05-15
```

**Arguments:**
- `--dates`: Dates to download (YYYY-MM-DD).
- `--csv`: Target dates CSV with Year/Month/Day columns.
- `--csv-type`: Optional Type filter for `--csv` (e.g., Active).
- `--neutral-csv`: Optional second CSV to merge dates from.
- `--neutral-type`: Optional Type filter for `--neutral-csv` (e.g., Neutral).
- `--phenomenon`: Optional phenomenon filter for CSVs (e.g., AO).
- `--cache-dir`: Cache directory (default: `/scratch-shared/ekasteleyn/downloads/hres_t0`).
- `--include-prev-day`: Also download the previous day for each date (needed for `init_hour=0`).

### 2. `extract_latents_hres.py`
This script integrates HRES T0 data loading with the Aurora model and runs inference to extract latent representations and attention weights from specified model layers (Perceiver and Encoder layers).

**Features:**
- Runs predictions and extracts latents/attention weights using PyTorch forward hooks.
- Saves latents as `.pt` files and predictions as `.nc` (NetCDF) files.
- Thread-pool based asynchronous saving to disk and optionally to an S3 bucket (if credentials are provided in `.env`).
- Supports specifying both 00:00 and 12:00 UTC initialization hours.

**Usage:**
```bash
python extract_latents_hres.py --dates 2022-01-15 --num-steps 1 --init-hours 12 --save-predictions
```

**Arguments:**
- `--dates`: Dates to process (YYYY-MM-DD).
- `--dates-csv`: Path to CSV file with 'date' column or 'Year', 'Month', 'Day'.
- `--num-steps`: Number of rollout steps (default: 1).
- `--output-dir`: Output directory (default: `/tmp/$USER/aurora_hres_latents`).
- `--cache-dir`: Cache directory for downloads.
- `--save-predictions`: Also save prediction outputs as NetCDF.
- `--init-hours`: Init hours to process (0 and/or 12). Default: `[12]`.
- `--latent-init-hour`: Only extract latents for this init hour.
- `--compile`: Use `torch.compile()` for faster inference (alters accumulation order).
- `--fp64`: Run model and save predictions in float64.

### 3. `run_aurora_hres_official.py`
A validation script that follows the official Microsoft Aurora example exactly. It runs the model on three predefined dates in 2022 and compares the predictions against the official WeatherBench2 Aurora benchmark.

**Features:**
- Downloads data for predefined dates.
- Formats the batch explicitly for the 12:00 UTC initialization.
- Generates predictions up to 3 steps (18 hours lead time).
- Computes latitude-weighted Root Mean Square Error (RMSE) for Geopotential and Temperature at 500, 700, and 850 hPa levels.
- Outputs a comparison CSV with the benchmark results.

**Usage:**
```bash
python run_aurora_hres_official.py --dates 2022-01-15 --num-steps 3
```

**Arguments:**
- `--dates`: Dates to run predictions for (default: `['2022-01-15', '2022-05-15', '2022-09-15']`).
- `--num-steps`: Number of rollout steps (default: 3).

## Requirements
- `xarray`
- `torch`
- `aurora`
- `huggingface_hub`
- `fsspec`
- `pandas`
- `numpy`
- `boto3` (optional, for S3 uploads)
