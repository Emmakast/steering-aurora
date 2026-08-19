#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def main():
    multi_csv = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)/ao_indices.csv'
    single_csv = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_inject/ao_indices.csv'
    out_path = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/injection_ablation_table.png'

    multi_df = pd.read_csv(multi_csv)
    single_df = pd.read_csv(single_csv)

    alphas = sorted(multi_df['Alpha'].unique())

    # Extract AO_Index_Corrected
    multi_vals = {row['Alpha']: row['AO_Index_Corrected'] for _, row in multi_df.iterrows()}
    single_vals = {row['Alpha']: row['AO_Index_Corrected'] for _, row in single_df.iterrows()}

    base_val = multi_vals[0.0]

    # Build table data
    table_data = []
    for a in alphas:
        table_data.append([f"{multi_vals[a]:.4f}", f"{single_vals[a]:.4f}"])

    col_labels = ["Multi", "Single"]
    row_labels = [f"α = {a}" for a in alphas]

    fig, ax = plt.subplots(figsize=(3, 5))
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    # We want to map colors based on the difference from base_val
    all_diffs = [multi_vals[a] - base_val for a in alphas] + [single_vals[a] - base_val for a in alphas]
    max_diff = max(all_diffs)
    min_diff = min(all_diffs)

    # RdBu: Red is low (negative), Blue is high (positive)
    cmap = cm.RdBu
    # Center the colormap at 0 difference
    norm = mcolors.TwoSlopeNorm(vmin=min_diff if min_diff < 0 else -0.1, vcenter=0.0, vmax=max_diff if max_diff > 0 else 0.1)

    for (row, col), cell in table.get_celld().items():
        if row == 0 and col >= 0:
            # Column headers
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e1e6ed')
        elif col == -1:
            # Row labels
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e1e6ed')
        elif row > 0 and col >= 0:
            # Data cells
            alpha = alphas[row-1]
            val = float(table_data[row-1][col])
            diff = val - base_val
            
            # Get color from colormap based on difference
            color = cmap(norm(diff))
            # Soften color to ensure text is readable
            color = tuple(c*0.6 + 0.4 for c in color[:3]) + (1.0,)
            cell.set_facecolor(color)
            
            # Emphasize alpha 0 row
            if alpha == 0.0:
                cell.set_text_props(weight='bold')

    fig.canvas.draw()
    bbox = table.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_path, dpi=300, bbox_inches=bbox.expanded(1.05, 1.05), facecolor='white')
    print(f"Saved table to {out_path}")

if __name__ == "__main__":
    main()
