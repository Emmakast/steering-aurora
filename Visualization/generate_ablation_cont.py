#!/usr/bin/env python3
"""
Generate an ablation study figure for the AO contrastive pairs.
Plots a 2x5 grid:
- Col 0: Ground Truth (Base) + Table
- Col 1: N=1
- Col 2: N=10
- Col 3: N=232
- Col 4: N=81
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

    ax.set_title(title, fontsize=42, fontweight="bold", pad=12)
    return im

def main():
    base_dir = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors")
    date = "20170308"
    
    configs = [
        {"label": "N=1", "dir": "AO_1encoder(2)_cont", "file": f"steered_ao_cont_1_{date}_1200_polar_both_lat60p0_alpha_5.0.nc"},
        {"label": "N=10", "dir": "AO_1encoder(2)_cont10", "file": f"steered_ao_cont_10_{date}_1200_polar_both_lat60p0_alpha_5.0.nc"},
        {"label": "N=232", "dir": "AO_1encoder(2)_cont232", "file": f"steered_ao_cont_232_{date}_1200_polar_both_lat60p0_alpha_5.0.nc"},
        {"label": "N=81", "dir": "AO_1encoder(2)", "file": f"steered_ao_ao81_polar_{date}_1200_polar_north_lat60p0_alpha_5.0.nc"},
    ]

    # Base file (assume same for all, load from normal)
    base_file = base_dir / "AO_1encoder(2)" / f"base_ao_ao81_polar_{date}_1200_alpha_0.0.nc"
    base_ds = xr.open_dataset(base_file)
    
    lat = base_ds.latitude.values
    lon = base_ds.longitude.values
    lons2d, lats2d = np.meshgrid(lon, lat)
    
    base_field = _extract_field(base_ds, "z", 50)
    
    data_crs = ccrs.PlateCarree()
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    extent = [-180, 180, 20, 90]

    steered_fields = []
    diff_fields = []
    
    # Read table data and numerical values for color-coding
    unified_csv = Path("/home/ekasteleyn/aurora_thesis/thesis/results/all_indices_evaluated.csv")
    unified_df = pd.read_csv(unified_csv) if unified_csv.exists() else None
    
    base_index_val = "N/A"
    base_num = np.nan
    if unified_df is not None:
        # Match base file
        base_row = unified_df[(unified_df["Alpha"] == 0.0) & (unified_df["Filename"] == f"base_ao_ao81_polar_{date}_1200_alpha_0.0.nc")]
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
            
        idx_val = "N/A"
        num_val = np.nan
        found = False
        if unified_df is not None:
            steered_row = unified_df[unified_df["Filename"] == cfg["file"]]
            if not steered_row.empty:
                num_val = steered_row.iloc[0]["AO_Index_Corrected"]
                idx_val = f'{num_val:.4f}'
                found = True
                
        if not found:
            local_csv = base_dir / cfg["dir"] / "all_indices_evaluated.csv"
            if local_csv.exists():
                local_df = pd.read_csv(local_csv)
                steered_row = local_df[local_df["Filename"] == cfg["file"]]
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
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(
        3, 5, figure=fig,
        height_ratios=[1, 1, 0.05],
        hspace=0.15, wspace=0.05,
        left=0.02, right=0.98, top=0.90, bottom=0.12
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
        colLabels=["N", "AO Index"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(28)
    table.scale(1.0, 3.5)
    
    # Calculate color coding based on diff from base
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    valid_diffs_list = [val - base_num for val in index_nums if not np.isnan(val)]
    max_diff = max(valid_diffs_list) if valid_diffs_list and max(valid_diffs_list) > 0 else 0.1
    min_diff = min(valid_diffs_list) if valid_diffs_list and min(valid_diffs_list) < 0 else -0.1
    
    cmap = cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=min_diff, vcenter=0.0, vmax=max_diff)

    # Apply bold header and color coding
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        elif col == 1:
            val = index_nums[row - 1]
            if not np.isnan(val):
                diff = val - base_num
                color = cmap(norm(diff))
                # Mix with white to soften (70% white) to avoid deep navy looking gray
                color = tuple(c*0.3 + 0.7 for c in color[:3]) + (1.0,)
                cell.set_facecolor(color)

    # ── Col 1-4: Encoders ──
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
    cbar_ax1 = fig.add_subplot(gs[2, 1:3])
    cb_raw = fig.colorbar(im_raw, cax=cbar_ax1, orientation="horizontal")
    cb_raw.set_label("Geopotential Height (50 hPa) [m² s⁻²]", fontsize=38, fontweight="bold")
    cb_raw.ax.tick_params(labelsize=32)

    cbar_ax2 = fig.add_subplot(gs[2, 3:])
    cb_diff = fig.colorbar(im_diff, cax=cbar_ax2, orientation="horizontal")
    cb_diff.set_label("Diff [m² s⁻²]", fontsize=38, fontweight="bold")
    cb_diff.ax.tick_params(labelsize=32)

    out_path = base_dir / "ablation_ao_contrastive_unified.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved ablation figure to {out_path}")

if __name__ == "__main__":
    main()
