# Aurora Model Steering

This directory contains the core scripts for implementing Contrastive Activation Addition (CAA) steering on the Microsoft Aurora model. These scripts compute steering vectors from previously extracted latent representations and inject them back into the model during inference to physically steer meteorological phenomena.

## Files

### 1. `steer_aurora.py`
The standard single-layer steering pipeline. It extracts the difference between the mean "Active" and mean "Neutral" latent states for a given climate phenomenon, and injects this difference vector into a target encoder layer.

**Features:**
- Computes steering vectors dynamically from pre-extracted latents (downloads from S3 if not available locally).
- Supports spatial masking: restricts the steering vector to specific geographical regions like `polar` (with adjustable minimum latitude and hemisphere) or `tropical` (with adjustable maximum latitude).
- Applies PyTorch forward hooks to inject the scaled steering vector (controlled by `alpha`) at each step of the rollout.
- Saves base (unsteered) and steered predictions as NetCDF files.

**Usage:**
```bash
python steer_aurora.py --phenomenon AO --alphas 0.5 1.0 --encoder-idx 2 --mask-region polar --polar-lat-min 60.0
```

**Key Arguments:**
- `--alphas`: List of steering strengths.
- `--phenomenon`: Phenomenon to steer (e.g., AO, MJO, ENSO).
- `--csv`: Target dates CSV file.
- `--encoder-idx`: Encoder index to steer (0, 1, or 2). Default is `2`.
- `--mask-region`: Spatial mask region (`none`, `polar`, `tropical`).
- `--polar-lat-min` / `--tropical-lat-max`: Latitude boundaries for spatial masks.

### 2. `steer_aurora_ablated.py`
An experimental variant of the steering pipeline that applies ablation to the latent channels or vertical Z-levels before injection. This is used to understand which parts of the latent space (e.g., specific vertical levels or channel chunks) encode the physical properties of the phenomenon.

**Features:**
- Includes all standard steering features (spatial masking, CSV loading, hook injection).
- Applies a Z-index or Channel-slice ablation:
  - For 5D latents (Swin backbone): Isolates a specific Z-level and zeroes out the rest of the vector.
  - For 3D latents (Encoder output): Partitions the channels into 3 slices and steers only the selected slice.
  
**Usage:**
```bash
python steer_aurora_ablated.py --phenomenon AO --alphas 1.0 --latent-idx 1
```

**Key Arguments:**
- `--latent-idx`: The latent Z-index or channel slice index (0, 1, or 2) to isolate and steer.

### 3. `steer_aurora_multi.py`
A multi-layer steering variant. Instead of steering a single encoder layer, it computes steering vectors for multiple layers concurrently and injects them simultaneously during the forward pass.

**Features:**
- Computes and caches steering vectors for `encoder_0`, `encoder_1`, and `encoder_2`.
- Injects vectors into all three layers simultaneously using multiple forward hooks.
- Supports layer-specific weight factors to scale the relative strength of the intervention across different depths of the network.

**Usage:**
```bash
python steer_aurora_multi.py --phenomenon AO --alphas 1.0 2.0 --layer-weights 0.1 0.3 1.0
```

**Key Arguments:**
- `--layer-weights`: Layer-specific alpha weight factors (default: `0.1 0.3 1.0` for `encoder_0`, `encoder_1`, `encoder_2` respectively).

## Requirements
- `torch`
- `xarray`
- `pandas`
- `numpy`
- `aurora`
- `boto3` (optional, for fetching latents from S3)
- `python-dotenv`
