#!/usr/bin/env python3
"""
Extract latents for ONE day in both FP32 and FP16, then plot overlaid
distributions so we can visually compare precision loss.
"""
from __future__ import annotations

import gc
import os
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from aurora import Aurora, Batch, Metadata


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr"
    "/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)
STATIC_PATH = os.path.expanduser("~/downloads/era5/static.nc")
DATE = "2024-01-01"
OUTPUT_DIR = Path.home() / "aurora_thesis" / "thesis" / "results"

TARGET_LAYERS = [
    ("perceiver", 0),
    ("encoder", 0),
    ("encoder", 1),
    ("encoder", 2),
]

VAR_MAP = {
    "2t":  "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t":   "temperature",
    "u":   "u_component_of_wind",
    "v":   "v_component_of_wind",
    "q":   "specific_humidity",
    "z":   "geopotential",
}


# ═══════════════════════════════════════════════════════════════════════
# Data loading (from extract_latents.py)
# ═══════════════════════════════════════════════════════════════════════
def load_static(ds, static_path):
    ref = ds.isel(latitude=slice(0, 720), longitude=slice(0, 1440))
    static = xr.open_dataset(static_path, engine="netcdf4")
    if "valid_time" in static.dims:
        static = static.isel(valid_time=0)
    static = static.interp(latitude=ref.latitude, longitude=ref.longitude)
    static = static.transpose("latitude", "longitude")
    return {
        "z":   torch.from_numpy(static["z"].values).float(),
        "slt": torch.from_numpy(static["slt"].values).float(),
        "lsm": torch.from_numpy(static["lsm"].values).float(),
    }


def load_batch(ds, static_vars, date_str):
    target_time = pd.to_datetime(f"{date_str}T06:00:00")
    request_times = [target_time - timedelta(hours=6), target_time]
    frame = ds.sel(time=request_times, method="nearest").load()
    frame = frame.sortby("time").isel(latitude=slice(0, 720), longitude=slice(0, 1440))
    surf = {
        k: torch.from_numpy(
            frame[VAR_MAP[k]].transpose("time", "latitude", "longitude").values
        ).unsqueeze(0).float()
        for k in ["2t", "10u", "10v", "msl"]
    }
    atmos = {
        k: torch.from_numpy(
            frame[VAR_MAP[k]].transpose("time", "level", "latitude", "longitude").values
        ).unsqueeze(0).float()
        for k in ["t", "u", "v", "q", "z"]
    }
    return Batch(
        surf_vars=surf, static_vars=static_vars, atmos_vars=atmos,
        metadata=Metadata(
            lat=torch.from_numpy(frame.latitude.values),
            lon=torch.from_numpy(frame.longitude.values),
            time=tuple(pd.to_datetime(frame.time.values).to_pydatetime()),
            atmos_levels=tuple(int(l) for l in frame.level.values),
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# Hooks
# ═══════════════════════════════════════════════════════════════════════
def register_hooks(model, target_layers):
    activations = {}
    handles = []
    for part, idx in target_layers:
        key = f"{part}_{idx}"

        def _make_hook(k):
            def _hook(_module, _input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                activations[k] = tensor.detach().cpu()
            return _hook

        if part == "perceiver":
            handles.append(model.encoder.register_forward_hook(_make_hook(key)))
        elif part == "encoder":
            handles.append(model.backbone.encoder_layers[idx].register_forward_hook(_make_hook(key)))
    return activations, handles


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════
def plot_fp32_vs_fp16(tensors_fp32: dict[str, torch.Tensor]):
    n = len(tensors_fp32)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]

    for i, (name, t32) in enumerate(tensors_fp32.items()):
        t16 = t32.half()
        v32 = t32.flatten().numpy()
        v16 = t16.float().flatten().numpy()
        err = np.abs(v32 - v16)

        # --- Col 0: overlaid histograms ---
        ax = axes[i, 0]
        lo, hi = np.percentile(v32, [0.5, 99.5])
        bins = np.linspace(lo, hi, 600)
        ax.hist(v32, bins=bins, alpha=0.6, color="steelblue", label="FP32",
                density=True, edgecolor="none")
        ax.hist(v16, bins=bins, alpha=0.5, color="coral", label="FP16",
                density=True, edgecolor="none")
        ax.set_title(f"{name} — distribution", fontsize=11)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        # --- Col 1: zoomed-in overlay (narrow window around mean) ---
        ax = axes[i, 1]
        mu = np.mean(v32)
        sigma = np.std(v32)
        zlo, zhi = mu - 0.3 * sigma, mu + 0.3 * sigma
        zbins = np.linspace(zlo, zhi, 500)
        ax.hist(v32, bins=zbins, alpha=0.6, color="steelblue", label="FP32",
                density=True, edgecolor="none")
        ax.hist(v16, bins=zbins, alpha=0.5, color="coral", label="FP16",
                density=True, edgecolor="none")
        ax.set_title(f"{name} — zoomed (±0.3σ around mean)", fontsize=11)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        # --- Col 2: absolute error distribution ---
        ax = axes[i, 2]
        err_finite = err[err > 0]
        if len(err_finite) > 0:
            ax.hist(np.log10(err_finite), bins=300, alpha=0.7,
                    color="mediumpurple", edgecolor="none", density=True)
        ax.set_title(f"{name} — |FP32 − FP16| error", fontsize=11)
        ax.set_xlabel("log₁₀(absolute error)")
        ax.set_ylabel("Density")
        ax.text(0.98, 0.95,
                f"mean err = {np.mean(err):.2e}\n"
                f"max err  = {np.max(err):.2e}\n"
                f"median err = {np.median(err):.2e}\n"
                f"% exact = {100 * np.mean(err == 0):.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    fig.suptitle(f"FP32 vs FP16 Precision — {DATE}", fontsize=14, y=1.01)
    out = OUTPUT_DIR / f"fp32_vs_fp16_{DATE}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening Zarr …")
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)

    print("Loading static variables …")
    static_vars = load_static(ds, STATIC_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model …")
    model = Aurora()
    model.load_checkpoint()
    model.eval().to(device)

    activations, handles = register_hooks(model, TARGET_LAYERS)

    print(f"Loading batch for {DATE} …")
    batch = load_batch(ds, static_vars, DATE).to(device)

    print("Forward pass …")
    with torch.inference_mode():
        model(batch)

    # Collect FP32 tensors
    tensors_fp32 = {}
    for key, tensor in activations.items():
        tensors_fp32[key] = tensor.float()
        print(f"  {key}: shape={tuple(tensor.shape)} dtype=float32")

    for h in handles:
        h.remove()
    del batch, model
    torch.cuda.empty_cache()
    gc.collect()

    print("\nPlotting FP32 vs FP16 …")
    plot_fp32_vs_fp16(tensors_fp32)


if __name__ == "__main__":
    main()
