# Prior Analysis

This directory contains scripts used for foundational analysis and validation of the Microsoft Aurora model before conducting latent steering experiments. The primary goals are to verify reproducibility against official baselines and to analyze precision errors when using lower-precision data types.

## Files

### 1. `compare_fp64_to_wb2.py`
Validates that the local Aurora model setup can perfectly reproduce the official WeatherBench2 (WB2) Aurora predictions. 

**Features:**
- Downloads initial IFS HRES T0 data from Google Cloud Storage (GCS).
- Runs the Aurora 0.25° Fine-Tuned model in double precision (FP64) to avoid floating-point accumulation discrepancies.
- Compares the outputs against WB2 Aurora predictions (surface variables and atmospheric variables at various pressure levels) for the same valid times.
- Reports statistics on exact bit-for-bit matches, max/mean absolute differences, and summarizes potential sources of discrepancy.
- Saves the comparison summary to a CSV file.

**Usage:**
```bash
python compare_fp64_to_wb2.py --date 2022-01-15 --init-hours 0 12 --device cuda
```

**Arguments:**
- `--date`: Date to test (YYYY-MM-DD). Default is `2022-01-15`.
- `--init-hours`: Initialization hours to test (default: `0 12`).
- `--device`: Target device (default: `cuda` if available, else `cpu`).
- `--fp32`: Run in FP32 instead of FP64.

### 2. `plot_fp32_vs_fp16.py`
Evaluates the precision loss introduced when internal model activations (latents) are cast from FP32 to FP16. This is crucial for verifying if extracting latents in FP16 (to save memory and PCIe bandwidth) introduces unacceptable distortion.

**Features:**
- Loads ERA5 input data for a single specified date from a local Zarr store.
- Runs an Aurora forward pass using PyTorch hooks to extract internal activations from target layers (`perceiver_0`, `encoder_0`, `encoder_1`, `encoder_2`) in full FP32 precision.
- Flattens and casts the latents to FP16 to compute precision errors.
- Generates a comprehensive plot containing:
  - Overlaid density histograms of FP32 vs. FP16 distributions.
  - Zoomed-in density plots centered around the mean (±0.3σ).
  - Histograms of log₁₀(absolute error) with summary statistics (mean error, max error, and % exact match).
- Saves the resulting composite plot as a PNG.

**Usage:**
```bash
python plot_fp32_vs_fp16.py
```

## Requirements
- `xarray`
- `torch`
- `pandas`
- `numpy`
- `matplotlib`
- `aurora`
- `huggingface_hub`
- `fsspec`
