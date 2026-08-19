#!/usr/bin/env python3
"""
Generate a line plot showing AO Index over the amount of rollout days.
Plots lines for specific alphas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    csv_path = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_multiroll/multiroll_ao_indices.csv")
    out_path = Path("/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_multiroll/multiroll_ao_index_plot.png")
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Extract step number from the 'Folder' column (e.g. 'steps_4' -> 4)
    # Assume 1 step = 6 hours -> 4 steps = 1 day
    df['Steps'] = df['Folder'].str.extract(r'steps_(\d+)').astype(int)
    df['Days'] = df['Steps'] / 4.0
    
    # We want to plot alphas 1, 2, 5, 10. Let's also include 0 (Base) for reference.
    target_alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    df_filtered = df[df['Alpha'].isin(target_alphas)].copy()
    
    # Sort by Days to ensure lines plot correctly
    df_filtered = df_filtered.sort_values(by=['Alpha', 'Days'])
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Custom colorblind palette so it is distinct and accessible
    cb_palette = sns.color_palette("colorblind")
    palette = {
        0.0: "black",
        1.0: cb_palette[0],  # Blue
        2.0: cb_palette[1],  # Orange
        5.0: cb_palette[2],  # Green
        10.0: cb_palette[3]  # Red
    }
    
    for alpha in target_alphas:
        subset = df_filtered[df_filtered['Alpha'] == alpha]
        label_str = "Base (α=0)" if alpha == 0.0 else f"Steered (α={int(alpha)})"
        
        plt.plot(
            subset['Days'], 
            subset['AO_Index_Corrected'], 
            marker='o', 
            linewidth=2.5, 
            markersize=8,
            color=palette[alpha],
            label=label_str
        )
        
    # Plot ERA5 reference if available
    era5_path = Path("/home/ekasteleyn/aurora_thesis/era5_ao_reference.csv")
    if era5_path.exists():
        era5_df = pd.read_csv(era5_path)
        # Data is 6-hourly starting from Day 0
        era5_df['Days'] = [i * 0.25 for i in range(len(era5_df))]
        plt.plot(
            era5_df['Days'], 
            era5_df['era5_ao'], 
            marker='o', 
            linewidth=3.0, 
            linestyle='--',
            markersize=8,
            color='gray',
            label='Actual ERA5 (Ground Truth)'
        )
        
    plt.xlabel("Forecast Lead Time (Days)", fontsize=16, fontweight="bold")
    plt.ylabel("AO Index", fontsize=16, fontweight="bold")
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Configuration", title_fontsize=15, fontsize=14, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
