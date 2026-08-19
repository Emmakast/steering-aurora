#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Load CSV
    csv_path = "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/81_eval_absolute.csv"
    out_path = "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/81_eval_summary_table.png"
    
    df = pd.read_csv(csv_path)

    table_data = []
    
    for _, row in df.iterrows():
        alpha = f"{row['Alpha']:.1f}"
        
        abs_mean = f"{row['Abs_Mean']:.4f}"
        abs_mean = f"+{abs_mean}" if row['Abs_Mean'] > 0 else abs_mean
        abs_std = f"±{row['Abs_Std']:.4f}"
        
        diff_mean = f"{row['Diff_Mean']:.4f}"
        diff_mean = f"+{diff_mean}" if row['Diff_Mean'] > 0 else diff_mean
        diff_std = f"±{row['Diff_Std']:.4f}"
        
        table_data.append([alpha, abs_mean, abs_std, diff_mean, diff_std])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")

    col_labels = ["α", "Mean Absolute", "Std Dev Absolute", "Mean Diff", "Std Dev Diff"]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    table.auto_set_column_width(list(range(len(col_labels))))

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.Blues
    
    # We want to color the Mean Diff column based on magnitude
    max_diff = df["Diff_Mean"].max()
    norm = mcolors.Normalize(vmin=0, vmax=max_diff)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#e1e6ed")
        elif col == 3:  # Mean Diff column
            val = df.iloc[row - 1]["Diff_Mean"]
            color = cmap(norm(val))
            # Soften color for readability
            color = tuple(min(1.0, c + 0.3) for c in color[:3]) + (1.0,)
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold')

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved table PNG to {out_path}")

if __name__ == "__main__":
    main()
