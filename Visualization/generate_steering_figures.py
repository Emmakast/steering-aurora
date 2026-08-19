#!/usr/bin/env python3
"""
generate_steering_figures.py
============================
Publication-quality visualization of representation-engineering (active steering)
in the Aurora foundation model.

Produces two figures:
  Figure 1 – Dose-Response Grid (2×3):  α = {-5, 0, +5}
      Row 1: primary variable (ENSO→msl, MJO→u850, AO→z1000, others→z500)
      Row 2: causal difference maps (cols 0,2) + dose-response line graph (col 1)

  Figure 2 – Physical Profile (2×3):  fixed α = +5
      Row 1: steered prediction for three variables (z@500, msl, q@850)
      Row 2: causal difference (steered – base) for each variable

Usage:
  python generate_steering_figures.py \\
      --phenomenon NAO \\
      --date 20200206 \\
      --mask-tag polar_north_lat30p0 \\
      --data-dir /scratch-shared/ekasteleyn/nao_steered \\
      --csv-path /home/ekasteleyn/aurora_thesis/thesis/results/all_indices_evaluated.csv
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ──────────────────────────────────────────────────────────────
# Configuration tables
# ──────────────────────────────────────────────────────────────

# Phenomenon → projection / extent
PROJ_CONFIG = {
    "AO":   {"proj_cls": ccrs.NorthPolarStereo, "proj_kw": {"central_longitude": 0}, "extent": [-180, 180, 20, 90],  "circle_lat": 60},
    "NAO":  {"proj_cls": ccrs.PlateCarree, "proj_kw": {}, "extent": [-90, 40, 20, 80],  "circle_lat": None},
    "PNA":  {"proj_cls": ccrs.PlateCarree, "proj_kw": {"central_longitude": 180}, "extent": [150, 310, 10, 80],  "circle_lat": None},
    "AAO":  {"proj_cls": ccrs.SouthPolarStereo, "proj_kw": {"central_longitude": 0}, "extent": [-180, 180, -90, -20],  "circle_lat": -60},
    "ENSO": {"proj_cls": ccrs.PlateCarree, "proj_kw": {"central_longitude": 180}, "extent": [-180, 180, -90, 90], "circle_lat": None},
    "MJO":  {"proj_cls": ccrs.PlateCarree, "proj_kw": {"central_longitude": 180}, "extent": [-180, 180, -90, 90], "circle_lat": None},
}

# Phenomenon → default date, name-suffix, mask-tag, data-dir
PHENOM_DEFAULTS = {
    "NAO":  {"date": "20200206", "name_suffix": "nao",                "mask_tag": "polar_north_lat30p0",  "data_dir": "/scratch-shared/ekasteleyn/nao_steered"},
    "PNA":  {"date": "20190826", "name_suffix": "pna",                "mask_tag": "nomask",               "data_dir": "/scratch-shared/ekasteleyn/pna_neutral_steered"},
    "AO":   {"date": "20160123", "name_suffix": "ao_ao81_polar",      "mask_tag": "polar_north_lat60p0",  "data_dir": "/scratch-shared/ekasteleyn/ao_neutral_steered"},
    "AAO":  {"date": "20160113", "name_suffix": "aao_aao_antarctic",  "mask_tag": "polar_south_lat60p0",  "data_dir": "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AAO_1encoder(2)"},
    "ENSO": {"date": "20170103", "name_suffix": "enso",               "mask_tag": "tropical_lat30p0",     "data_dir": "/home/ekasteleyn/aurora_thesis/thesis/results"},
    "MJO":  {"date": "20160123", "name_suffix": "mjo",                "mask_tag": "tropical_lat30p0",     "data_dir": "/home/ekasteleyn/aurora_thesis/thesis/results"},
}

# Phenomenon → target index column in the CSV
PHENOM_INDEX_COL = {
    "NAO": "NAO", "PNA": "PNA", "AO": "AO_Index_Corrected", "AAO": "AAO",
    "ENSO": "ENSO", "MJO": "MJO",
}

# Figure 1 primary variable per phenomenon  (var, level_or_None, label, unit)
FIG1_PRIMARY_VAR = {
    "ENSO": ("msl", None, "Mean Sea-Level Pressure",            "Pa"),
    "MJO":  ("u",   850,  "Zonal Wind (850 hPa)",               "m s⁻¹"),
    "AO":   ("z",   50,   "Geopotential Height (50 hPa)",     "m² s⁻²"),
    "AAO":  ("z",   500,  "Geopotential Height (500 hPa)",      "m² s⁻²"),
    "NAO":  ("z",   500,  "Geopotential Height (500 hPa)",      "m² s⁻²"),
    "PNA":  ("z",   500,  "Geopotential Height (500 hPa)",      "m² s⁻²"),
}

# Figure 2 variable definitions  (var_name, level_or_None, display_label, unit)
PROFILE_VARS = [
    ("z",   500,  "z at 500hPa",  "m² s⁻²"),
    ("msl", None, "MSLP",         "Pa"),
    ("q",   850,  "q at 850hPa",     "kg kg⁻¹"),
]


# ──────────────────────────────────────────────────────────────
# Helper: resolve file paths from naming convention
# ──────────────────────────────────────────────────────────────

def _find_base_file(data_dir: Path, name_suffix: str, date: str) -> Path:
    """Try two naming conventions for the base file."""
    init_tag = "1200"
    candidates = [
        data_dir / f"base_{name_suffix}_{date}_{init_tag}_alpha_0.0.nc",
        # fallback: just the phenomenon slug
        data_dir / f"base_{name_suffix.split('_')[0]}_{date}_{init_tag}_alpha_0.0.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Base file not found.  Tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def _find_steered_file(data_dir: Path, name_suffix: str, date: str,
                        mask_tag: str, alpha: float) -> Path | None:
    init_tag = "1200"
    if mask_tag:
        f = data_dir / f"steered_{name_suffix}_{date}_{init_tag}_{mask_tag}_alpha_{alpha}.nc"
    else:
        f = data_dir / f"steered_{name_suffix}_{date}_{init_tag}_alpha_{alpha}.nc"
    return f if f.exists() else None


# ──────────────────────────────────────────────────────────────
# Helper: extract a 2-D field from an xarray Dataset
# ──────────────────────────────────────────────────────────────

def _extract_field(ds: xr.Dataset, var: str, level: int | None = None) -> np.ndarray:
    da = ds[var]
    if level is not None and "level" in da.dims:
        da = da.sel(level=level)
    # squeeze any remaining singleton dims (e.g. time, batch)
    return da.squeeze().values


# ──────────────────────────────────────────────────────────────
# Helper: mask boundary drawing
# ──────────────────────────────────────────────────────────────

def _draw_mask_boundary(ax, proj_cfg, data_crs):
    """
    Draw the spatial mask boundary.
    Polar projections → circle at the given latitude.
    PlateCarree       → bounding box at ±30° latitude.
    """
    circle_lat = proj_cfg["circle_lat"]
    if circle_lat is not None:
        # Draw a circle at the specified latitude
        theta = np.linspace(0, 2 * np.pi, 360)
        # Project latitude circle onto map coordinates
        lons_circle = np.degrees(theta)
        lats_circle = np.full_like(lons_circle, circle_lat)
        ax.plot(
            lons_circle, lats_circle,
            transform=data_crs,
            color="black", linewidth=1.5, linestyle="--",
            zorder=5,
            path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()],
        )
    else:
        # PlateCarree → draw a bounding rectangle at ±30°
        lat_bound = 30
        lons_box = np.concatenate([
            np.linspace(-180, 180, 360),
            np.full(60, 180),
            np.linspace(180, -180, 360),
            np.full(60, -180),
        ])
        lats_box = np.concatenate([
            np.full(360, lat_bound),
            np.linspace(lat_bound, -lat_bound, 60),
            np.full(360, -lat_bound),
            np.linspace(-lat_bound, lat_bound, 60),
        ])
        ax.plot(
            lons_box, lats_box,
            transform=data_crs,
            color="black", linewidth=1.5, linestyle="--",
            zorder=5,
            path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()],
        )


# ──────────────────────────────────────────────────────────────
# Helper: polar extent (set_extent raises issues on polar stereo)
# ──────────────────────────────────────────────────────────────

def _set_map_extent(ax, proj_cfg, data_crs):
    extent = proj_cfg["extent"]
    try:
        ax.set_extent(extent, crs=data_crs)
    except Exception:
        ax.set_global()


# ──────────────────────────────────────────────────────────────
# Shared map-drawing routine
# ──────────────────────────────────────────────────────────────

def _draw_map(ax, lons2d, lats2d, field, cmap, vmin, vmax, title,
              proj_cfg, data_crs, draw_mask=False, extend="both", title_fontsize=14):
    _set_map_extent(ax, proj_cfg, data_crs)
    im = ax.pcolormesh(
        lons2d, lats2d, field,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=data_crs, rasterized=True, shading="auto",
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.4, edgecolor="#555555")
    # Gridlines
    gl_kw = dict(draw_labels=False, linewidth=0.3, color="grey", alpha=0.5, linestyle=":")
    try:
        ax.gridlines(**gl_kw)
    except Exception:
        pass

    if draw_mask:
        _draw_mask_boundary(ax, proj_cfg, data_crs)

    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=6)
    return im


# ──────────────────────────────────────────────────────────────
# Figure 1 – Dose-Response Grid
# ──────────────────────────────────────────────────────────────

def figure_dose_response(base_ds, steered_neg, steered_pos, csv_df,
                         phenomenon, date, name_suffix, proj_cfg, output_path, has_mask=True):
    """
    2×3 grid with GridSpec layout.
    Row 1: raw primary variable for α=-5, α=0, α=+5  +  colorbar row
    Row 2: diff(α=-5), dose-response line, diff(α=+5)  +  colorbar row
    """
    from matplotlib.gridspec import GridSpec

    data_crs = ccrs.PlateCarree()
    proj = proj_cfg["proj_cls"](**proj_cfg["proj_kw"])
    is_polar = proj_cfg["circle_lat"] is not None

    # Look up the primary variable for this phenomenon
    pvar, plevel, plabel, punit = FIG1_PRIMARY_VAR[phenomenon]

    lat = base_ds.latitude.values
    lon = base_ds.longitude.values
    lons2d, lats2d = np.meshgrid(lon, lat)

    base_field = _extract_field(base_ds, pvar, plevel)

    # Steered fields
    neg_field = _extract_field(steered_neg, pvar, plevel) if steered_neg else None
    pos_field = _extract_field(steered_pos, pvar, plevel) if steered_pos else None

    # Diffs
    diff_neg = (neg_field - base_field) if neg_field is not None else None
    diff_pos = (pos_field - base_field) if pos_field is not None else None

    # ── colour limits ──
    all_raw = [base_field]
    if neg_field is not None:
        all_raw.append(neg_field)
    if pos_field is not None:
        all_raw.append(pos_field)
    combined_raw = np.concatenate([f.ravel() for f in all_raw])
    vmin_raw = np.nanpercentile(combined_raw, 2)
    vmax_raw = np.nanpercentile(combined_raw, 98)

    diffs_for_lim = []
    if diff_neg is not None:
        diffs_for_lim.append(diff_neg)
    if diff_pos is not None:
        diffs_for_lim.append(diff_pos)
    if diffs_for_lim:
        max_diff = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs_for_lim])), 98)
    else:
        max_diff = 1.0

    # ── build figure with GridSpec ──
    # 4 rows: map_row1, cbar1, map_row2, cbar2
    fig_h = 18 if is_polar else 11.5
    fig = plt.figure(figsize=(15, fig_h))
    gs = GridSpec(
        6, 3, figure=fig,
        height_ratios=[1, 0.04, 0.20, 1, 0.10, 0.04],
        hspace=0.05, wspace=0.15,
        left=0.05, right=0.95, top=0.92, bottom=0.04,
    )

    # Row 1: three map axes
    ax_r1 = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
    # Row 2: two map axes + one Cartesian axis
    ax_r2_left  = fig.add_subplot(gs[3, 0], projection=proj)
    ax_r2_right = fig.add_subplot(gs[3, 2], projection=proj)
    ax_line     = fig.add_subplot(gs[3, 1])  # plain cartesian

    row1_axes = ax_r1
    row2_map_axes = [ax_r2_left, ax_r2_right]

    # ── Row 1: raw maps ──
    cmap_raw = "viridis"
    titles_r1 = [r"$\alpha = -5$", r"$\alpha = 0$ (Base)", r"$\alpha = +5$"]
    fields_r1 = [neg_field, base_field, pos_field]
    im_raw = None
    for ax, field, title in zip(row1_axes, fields_r1, titles_r1):
        if field is not None:
            im_raw = _draw_map(ax, lons2d, lats2d, field, cmap_raw,
                               vmin_raw, vmax_raw, title,
                               proj_cfg, data_crs, draw_mask=has_mask, title_fontsize=42)
        else:
            _set_map_extent(ax, proj_cfg, data_crs)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            ax.set_title(title + " (missing)", fontsize=42)

    # ── Row 2: diff maps ──
    cmap_diff = "RdBu_r"
    diff_labels = ["Difference", "Difference"]
    diff_fields = [diff_neg, diff_pos]
    im_diff = None
    for ax, field, title in zip(row2_map_axes, diff_fields, diff_labels):
        if field is not None:
            im_diff = _draw_map(ax, lons2d, lats2d, field, cmap_diff,
                                -max_diff, max_diff, title,
                                proj_cfg, data_crs, draw_mask=False, title_fontsize=42)
        else:
            _set_map_extent(ax, proj_cfg, data_crs)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            ax.set_title(title + " (missing)", fontsize=42)

    # ── Row 2 centre: dose-response line graph ──
    extent = proj_cfg["extent"]
    aspect = 1.0 if is_polar else (extent[3] - extent[2]) / (extent[1] - extent[0])
    _plot_dose_response_line(ax_line, csv_df, phenomenon, date, name_suffix, aspect=aspect)

    # ── Colorbars ──
    # Shared sequential colorbar below row 1
    if im_raw is not None:
        cbar_ax_raw = fig.add_subplot(gs[1, :])
        cb_raw = fig.colorbar(im_raw, cax=cbar_ax_raw, orientation="horizontal", extend="both")
        cb_raw.set_label(f"{plabel}  [{punit}]", fontsize=38)
        cb_raw.ax.tick_params(labelsize=34)

    # Shared diverging colorbar below row 2
    if im_diff is not None:
        cbar_ax_diff = fig.add_subplot(gs[5, :])
        cb_diff = fig.colorbar(im_diff, cax=cbar_ax_diff, orientation="horizontal", extend="both")
        cb_diff.set_label(f"Δ {plabel}  [{punit}]", fontsize=38)
        cb_diff.ax.tick_params(labelsize=34)

    # fig.suptitle removed

    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved Figure 1 → {output_path}")


# ──────────────────────────────────────────────────────────────
# Dose-response line graph (centre panel of Fig 1, row 2)
# ──────────────────────────────────────────────────────────────

def _plot_dose_response_line(ax, csv_df, phenomenon, date, name_suffix, aspect=0.5):
    """
    Plot target oscillation index vs α from the CSV.
    Highlight α ∈ {-5, 0, +5} with star markers and drop-lines.
    """
    idx_col = PHENOM_INDEX_COL[phenomenon]

    # Filter CSV rows for this phenomenon + date AND the correct name_suffix
    mask = csv_df["Filename"].str.contains(name_suffix, case=False)
    mask &= csv_df["Filename"].str.contains(date)
    df = csv_df.loc[mask].copy()

    if df.empty:
        ax.text(0.5, 0.5, "No CSV data found", transform=ax.transAxes,
                ha="center", va="center", fontsize=24, color="grey")
        ax.set_title("Response", fontsize=42, fontweight="bold")
        return

    df = df.sort_values("Alpha")

    if idx_col not in df.columns:
        fallback = f"{idx_col}_Index_Corrected"
        if fallback in df.columns:
            idx_col = fallback
        else:
            ax.text(0.5, 0.5, f"Column '{idx_col}' not found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=24, color="grey")
            ax.set_title("Response", fontsize=42, fontweight="bold")
            return

    alphas = df["Alpha"].values
    indices = df[idx_col].values

    # ── styling ──
    # All points as small dots
    ax.plot(alphas, indices, "o-", color="#5a7dba", markersize=5, linewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
            label=f"{phenomenon} index")

    # Highlight α = -5, 0, +5
    highlight_alphas = [-5.0, 0.0, 5.0]
    for ha in highlight_alphas:
        row = df.loc[df["Alpha"] == ha]
        if not row.empty:
            y_val = row[idx_col].values[0]
            ax.plot(ha, y_val, marker="*", color="#d4442e", markersize=16,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            # Drop-line to x-axis
            ax.vlines(ha, 0, y_val, colors="#d4442e", linestyles="dashed",
                      linewidth=0.8, alpha=0.6, zorder=2)

    # Baseline at y=0
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    # Force strictly linear x-axis from -10 to 10
    ax.set_xlim(-11, 11)
    ax.xaxis.set_major_locator(mticker.FixedLocator([-10, -5, 0, 5, 10]))
    ax.set_xlabel(r"Steering magnitude $\alpha$", fontsize=38)
    ax.set_ylabel("")
    ax.set_title("Response", fontsize=42, fontweight="bold", pad=6)
    ax.tick_params(labelsize=34)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Subtle background
    ax.set_facecolor("#f7f9fc")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # Set same aspect ratio as the map plots
    ax.set_box_aspect(aspect)


# ──────────────────────────────────────────────────────────────
# Figure 2 – Physical Profile
# ──────────────────────────────────────────────────────────────

def figure_physical_profile(base_ds, steered_ds, phenomenon, date,
                            proj_cfg, output_path):
    """
    2×3 grid with per-column colorbars, fixed α=+5.
    Row 1: steered prediction for z@500, msl, q@850
    Row 2: causal difference (steered – base) for each
    """
    from matplotlib.gridspec import GridSpec

    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree()
    is_polar = False
    
    local_proj_cfg = proj_cfg.copy()
    local_proj_cfg["extent"] = [-180, 180, -90, 90]
    local_proj_cfg["circle_lat"] = None

    lat = base_ds.latitude.values
    lon = base_ds.longitude.values
    lons2d, lats2d = np.meshgrid(lon, lat)

    # 5 rows: map_row1, cbar1, spacer, map_row2, cbar2
    fig_h = 11.5
    fig = plt.figure(figsize=(18, fig_h))
    gs = GridSpec(
        5, 3, figure=fig,
        height_ratios=[1, 0.05, 0.45, 1, 0.05],
        hspace=0.05, wspace=0.05,
        left=0.04, right=0.96, top=0.92, bottom=0.08,
    )

    # Map axes
    axes_r1 = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
    axes_r2 = [fig.add_subplot(gs[3, i], projection=proj) for i in range(3)]

    # Track colormaps for per-column colorbars
    im_row1_list = []
    im_row2_list = []

    for col_idx, (var, level, label, unit) in enumerate(PROFILE_VARS):
        # Check variable exists
        if var not in base_ds:
            for ax in [axes_r1[col_idx], axes_r2[col_idx]]:
                ax.set_title(f"{var} not found", fontsize=38)
                _set_map_extent(ax, local_proj_cfg, data_crs)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            continue

        base_field    = _extract_field(base_ds, var, level)
        steered_field = _extract_field(steered_ds, var, level) if steered_ds else None
        diff_field    = (steered_field - base_field) if steered_field is not None else None

        # Colour limits
        all_vals = [base_field]
        if steered_field is not None:
            all_vals.append(steered_field)
        combined = np.concatenate([v.ravel() for v in all_vals])
        vmin_r = np.nanpercentile(combined, 2)
        vmax_r = np.nanpercentile(combined, 98)

        if diff_field is not None:
            md = np.nanpercentile(np.abs(diff_field.ravel()), 98)
        else:
            md = 1.0

        # Row 1: steered prediction
        if steered_field is not None:
            im1 = _draw_map(axes_r1[col_idx], lons2d, lats2d, steered_field,
                            "viridis", vmin_r, vmax_r,
                            f"{label}",
                            local_proj_cfg, data_crs, draw_mask=False, title_fontsize=38)
            im_row1_list.append((im1, col_idx, unit))
        else:
            _set_map_extent(axes_r1[col_idx], local_proj_cfg, data_crs)
            axes_r1[col_idx].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            axes_r1[col_idx].set_title(f"{label}\n(missing)", fontsize=38)

        # Row 2: difference
        if diff_field is not None:
            im2 = _draw_map(axes_r2[col_idx], lons2d, lats2d, diff_field,
                            "RdBu_r", -md, md,
                            "Difference",
                            local_proj_cfg, data_crs, draw_mask=False, title_fontsize=38)
            im_row2_list.append((im2, col_idx, unit))
        else:
            _set_map_extent(axes_r2[col_idx], local_proj_cfg, data_crs)
            axes_r2[col_idx].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            axes_r2[col_idx].set_title("Difference\n(missing)", fontsize=38)

    # ── Per-column colorbars using GridSpec sub-axes ──
    for im, col_idx, unit in im_row1_list:
        cbar_ax = fig.add_subplot(gs[1, col_idx])
        cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
        cb.ax.tick_params(labelsize=30)
        cb.set_label(f"[{unit}]", fontsize=34)

    for im, col_idx, unit in im_row2_list:
        cbar_ax = fig.add_subplot(gs[4, col_idx])
        cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
        cb.ax.tick_params(labelsize=30)
        cb.set_label(f"Δ [{unit}]", fontsize=34)

    # fig.suptitle removed

    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved Figure 2 → {output_path}")


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality steering analysis figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phenomenon", type=str, required=True,
        choices=["AO", "NAO", "PNA", "AAO", "ENSO", "MJO"],
        help="Climatic phenomenon to visualise.",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Initialisation date tag, e.g. 20200206.  "
             "Defaults to the canonical date for each phenomenon.",
    )
    parser.add_argument(
        "--mask-tag", type=str, default=None,
        help="Spatial-mask tag in steered filenames (e.g. polar_north_lat30p0).  "
             "Defaults to the canonical tag for each phenomenon.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing the NetCDF files.  "
             "Defaults to the canonical directory for each phenomenon.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save the figures (defaults to data-dir).",
    )
    parser.add_argument(
        "--csv-path", type=str,
        default="/home/ekasteleyn/aurora_thesis/thesis/results/all_indices_evaluated.csv",
        help="Path to the master CSV with evaluated oscillation indices.",
    )
    parser.add_argument(
        "--name-suffix", type=str, default=None,
        help="Override the filename stem (e.g. 'ao_ao81_polar').  "
             "Defaults to the canonical suffix for each phenomenon.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output PNGs.  Defaults to data-dir.",
    )
    args = parser.parse_args()

    phenom = args.phenomenon
    defaults = PHENOM_DEFAULTS[phenom]

    date       = args.date        or defaults["date"]
    mask_tag   = args.mask_tag    or defaults["mask_tag"]
    name_suffix = args.name_suffix or defaults["name_suffix"]
    data_dir   = Path(args.data_dir or defaults["data_dir"])
    out_dir    = Path(args.out_dir) if args.out_dir else data_dir
    csv_path   = Path(args.csv_path)

    proj_cfg = PROJ_CONFIG[phenom]

    # ── Load CSV ──
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path)
    else:
        print(f"⚠ CSV not found at {csv_path}; dose-response panel will be empty.")
        csv_df = pd.DataFrame()

    # ── Load base dataset ──
    print(f"Loading base file for {phenom} / {date} …")
    base_file = _find_base_file(data_dir, name_suffix, date)
    base_ds = xr.open_dataset(base_file)
    print(f"  Base: {base_file.name}")

    # ── Load steered datasets for Figure 1 ──
    steered_neg_file = _find_steered_file(data_dir, name_suffix, date, mask_tag, -5.0)
    steered_pos_file = _find_steered_file(data_dir, name_suffix, date, mask_tag,  5.0)

    steered_neg = xr.open_dataset(steered_neg_file) if steered_neg_file else None
    steered_pos = xr.open_dataset(steered_pos_file) if steered_pos_file else None

    if steered_neg_file:
        print(f"  α=-5: {steered_neg_file.name}")
    else:
        print("  ⚠ α=-5 file not found – left column will be empty.")
    if steered_pos_file:
        print(f"  α=+5: {steered_pos_file.name}")
    else:
        print("  ⚠ α=+5 file not found – right column will be empty.")

    # ── Figure 1 ──
    fig1_out = out_dir / f"fig1_dose_response_{phenom.lower()}_{date}.png"
    has_mask = mask_tag != "nomask"
    figure_dose_response(
        base_ds, steered_neg, steered_pos,
        csv_df, phenom, date, name_suffix, proj_cfg, fig1_out, has_mask=has_mask
    )

    # ── Figure 2 (uses α=+5) ──
    steered_profile = steered_pos  # reuse the +5 dataset
    if steered_profile is not None:
        out2 = out_dir / f"fig2_physical_profile_{phenom.lower()}_{date}.png"
        figure_physical_profile(
            base_ds, steered_profile, phenom, date, proj_cfg, out2,
        )
    else:
        print("⚠ Cannot generate Figure 2 without α=+5 steered data.")

    # ── Clean up ──
    base_ds.close()
    if steered_neg is not None:
        steered_neg.close()
    if steered_pos is not None:
        steered_pos.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
