#!/usr/bin/env python3
"""
Generate an ablation study figure for the AO encoders.
Plots a 2x8 grid:
- Col 0: Ground Truth (Base) + Table of AO Indices
- Col 1-7: Encoders
- Top row: predictions (steered z50) for alpha=5
- Bottom row: causal differences (steered - base)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

def _extract_field(ds, var, level=None):
    da = ds[var]
    if level is not None and "level" in da.dims:
        da = da.sel(level=level)
    return da.squeeze().values

def _set_map_extent(ax, extent, data_crs):
    try:
        ax.set_extent(extent, crs=data_crs)
    except Exception:
        ax.set_global()

def _draw_map(ax, lons2d, lats2d, field, cmap, vmin, vmax, title, extent, data_crs):
    _set_map_extent(ax, extent, data_crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    im = ax.pcolormesh(lons2d, lats2d, field, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=data_crs, shading="nearest")
    
    # Draw mask circle at 60N
    theta = np.linspace(0, 2 * np.pi, 360)
    lons_circle = np.degrees(theta)
    lats_circle = np.full_like(lons_circle, 60)
    ax.plot(
        lons_circle, lats_circle,
        transform=data_crs,
        color="black", linewidth=1.5, linestyle="--",
        zorder=5,
        path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()],
    )

    gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.4, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False

    ax.set_title(title, fontsize=20, fontweight="bold", pad=8)
    return im

def main():
    base_dir = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors")
    date = "20170308"
    
    configs = [
        {"label": "Encoder 0", "dir": "AO_1encoder(0)", "file": f"steered_ao_ao81_polar_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
        {"label": "Encoder 1", "dir": "AO_1encoder(1)", "file": f"steered_ao_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
        {"label": "Encoder 2", "dir": "AO_1encoder(2)", "file": f"steered_ao_ao81_polar_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
        {"label": "Encoders (0,1)", "dir": "AO_2encoders(0,1)", "file": f"steered_ao_ao81_polar_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
        {"label": "Encoders (0,2)", "dir": "AO_2encoders(0,2)", "file": f"steered_ao_ao81_polar_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
        {"label": "Encoders (1,2)", "dir": "AO_2encoders(1,2)", "file": f"steered_ao_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
        {"label": "All 3 Encoders", "dir": "AO_3encoders", "file": f"steered_ao_ao81_polar_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
    ]

    # Base file (assume same for all, load from encoder 0)
    base_file = base_dir / "AO_1encoder(0)" / f"base_ao_ao81_polar_{date}_1200_alpha_0.0.nc"
    base_ds = xr.open_dataset(base_file)
    
    lat = base_ds.latitude.values
    lon = base_ds.longitude.values
    lons2d, lats2d = np.meshgrid(lon, lat)
    
    base_field = _extract_field(base_ds, "z", 50)
    
    data_crs = ccrs.PlateCarree()
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    extent = [-180, 180, 20, 90]

    # Store fields and indices
    steered_fields = []
    diff_fields = []
    
    # Read table data and numerical values for color-coding
    base_csv = base_dir / "AO_1encoder(0)" / "ao_indices.csv"
    base_index_val = "N/A"
    base_num = np.nan
    if base_csv.exists():
        df = pd.read_csv(base_csv)
        base_row = df[df["Alpha"] == 0.0]
        if not base_row.empty:
            base_num = base_row.iloc[0]["AO_Index_Corrected"]
            base_index_val = f'{base_num:.4f}'
            
    table_data = [["Base", base_index_val]]
    index_nums = [base_num]

    for cfg in configs:
        file_path = base_dir / cfg["dir"] / cfg["file"]
        if file_path.exists():
            ds = xr.open_dataset(file_path)
            field = _extract_field(ds, "z", 50)
            steered_fields.append(field)
            diff_fields.append(field - base_field)
            ds.close()
        else:
            print(f"Warning: File not found: {file_path}")
            steered_fields.append(np.zeros_like(base_field))
            diff_fields.append(np.zeros_like(base_field))
            
        csv_path = base_dir / cfg["dir"] / "ao_indices.csv"
        idx_val = "N/A"
        num_val = np.nan
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            steered_row = df[df["Alpha"] == 5.0]
            if not steered_row.empty:
                num_val = steered_row.iloc[0]["AO_Index_Corrected"]
                idx_val = f'{num_val:.4f}'
        table_data.append([cfg["label"], idx_val])
        index_nums.append(num_val)

    # Determine global min/max for colorbars
    valid_steered = [f for f in steered_fields if np.any(f)]
    valid_diffs = [f for f in diff_fields if np.any(f)]
    
    if valid_steered:
        vmin_raw = min(f.min() for f in valid_steered + [base_field])
        vmax_raw = max(f.max() for f in valid_steered + [base_field])
    else:
        vmin_raw, vmax_raw = 0, 1
        
    if valid_diffs:
        vmax_diff = max(np.abs(f).max() for f in valid_diffs)
        vmin_diff = -vmax_diff
    else:
        vmin_diff, vmax_diff = -1, 1

    # Plot
    fig = plt.figure(figsize=(26, 8))
    gs = GridSpec(
        3, 8, figure=fig,
        height_ratios=[1, 1, 0.05],
        hspace=0.1, wspace=0.05,
        left=0.02, right=0.98, top=0.95, bottom=0.08
    )

    im_raw = None
    im_diff = None
    
    # ── Col 0: Base & Table ──
    ax_base = fig.add_subplot(gs[0, 0], projection=proj)
    im_raw = _draw_map(ax_base, lons2d, lats2d, base_field, "viridis",
                       vmin_raw, vmax_raw, "Base", extent, data_crs)
                 
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis("off")
                  
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Configuration", "AO Index"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.0, 1.8) # Narrower width
    
    # Calculate color coding based on diff from base
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    valid_diffs = [abs(val - base_num) for val in index_nums if not np.isnan(val)]
    max_diff = max(valid_diffs) if valid_diffs and max(valid_diffs) > 0 else 1.0
    
    cmap = cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=max_diff)

    # Apply bold header and color coding
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        elif col == 1: # Color the AO Index column based on steering magnitude
            val = index_nums[row - 1]
            if not np.isnan(val):
                diff = abs(val - base_num)
                color = cmap(norm(diff))
                # Soften the deepest blue so text remains highly readable
                color = tuple(min(1.0, c + 0.35) for c in color[:3]) + (1.0,)
                cell.set_facecolor(color)

    # ── Col 1-7: Encoders ──
    for i, cfg in enumerate(configs):
        col = i + 1
        # Top row: Raw
        ax_raw = fig.add_subplot(gs[0, col], projection=proj)
        _draw_map(ax_raw, lons2d, lats2d, steered_fields[i], "viridis",
                  vmin_raw, vmax_raw, cfg["label"], extent, data_crs)

        # Bottom row: Diff
        ax_diff = fig.add_subplot(gs[1, col], projection=proj)
        im_diff = _draw_map(ax_diff, lons2d, lats2d, diff_fields[i], "RdBu_r",
                            vmin_diff, vmax_diff, "", extent, data_crs)

    # Colorbars
    cbar_ax1 = fig.add_subplot(gs[2, 1:4])
    cb_raw = fig.colorbar(im_raw, cax=cbar_ax1, orientation="horizontal")
    cb_raw.set_label("Geopotential Height (50 hPa) [m² s⁻²]", fontsize=18, fontweight="bold")
    cb_raw.ax.tick_params(labelsize=14)

    cbar_ax2 = fig.add_subplot(gs[2, 5:])
    cb_diff = fig.colorbar(im_diff, cax=cbar_ax2, orientation="horizontal")
    cb_diff.set_label("Difference [m² s⁻²]", fontsize=18, fontweight="bold")
    cb_diff.ax.tick_params(labelsize=14)

    out_path = base_dir / "ablation_ao_encoders.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved ablation figure to {out_path}")

if __name__ == "__main__":
    main()
