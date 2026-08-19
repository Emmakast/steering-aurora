# Visualization Scripts

This directory contains Python scripts designed to generate high-quality figures, maps, and tables for evaluating and analyzing the outcomes of the Aurora model steering experiments. These visual outputs compare base (unsteered) predictions with steered predictions across various experimental configurations.

## Files

### 1. Main Steering Visualizations
- **`generate_steering_figures.py`**
  The primary plotting script. Generates comprehensive map-based visualizations of the steering outcomes, plotting raw geopotential heights or other meteorological variables alongside the causal differences (steered minus base) to visualize the physical impact of the injected steering vectors.

### 2. Ablation Studies
- **`generate_ablation_figure.py`**
  Visualizes the impact of steering different combinations of encoder layers (e.g., Encoder 0 vs Encoder 2 vs all three). It produces a large grid plot comparing spatial maps, alongside a dynamically color-coded table displaying the resulting climatic index values (e.g., AO Index).
- **`generate_ablation_cont.py`**
  Evaluates the sensitivity of the steering vector to the number of contrastive pairs (`N`) used during extraction (e.g., `N=1`, `N=10`, `N=81`, `N=232`). Outputs a similar grid of maps and tables.
- **`generate_inject_ablation_table.py`**
  Generates a summarized color-coded table (`injection_ablation_table.png`) directly comparing Multi-layer injection performance versus Single-layer injection performance over a sweep of $\alpha$ values.

### 3. Evaluation and Timeseries
- **`generate_cross_index_plot.py`**
  Creates a multi-panel line plot (`cross_index_evaluation.png`) demonstrating "entanglement" or cross-talk between indices. It shows how steering one specific phenomenon (e.g., AO) inadvertently or intentionally impacts the measured indices of other phenomena (NAO, PNA, AAO, ENSO, MJO) across increasing steering magnitudes ($\alpha$).
- **`generate_multiroll_figure.py`**
  Visualizes the persistence of the steering effect over time. It generates a line plot (`multiroll_ao_index_plot.png`) showing how the target index evolves over several days of continuous forecast rollout for different $\alpha$ values, including a comparison against ERA5 ground truth data.
- **`generate_eval_table.py`**
  Creates a standalone summary table (`81_eval_summary_table.png`) that formats and color-codes mean absolute values and standard deviations of the evaluations across differing $\alpha$ values.

## Dependencies
These scripts heavily rely on Earth science and plotting libraries:
- `matplotlib` & `seaborn` (for plotting and tables)
- `cartopy` (for map projections, coastlines, and gridlines)
- `xarray` (for processing NetCDF files)
- `pandas` & `numpy` (for data manipulation)

## Usage
The scripts are generally run directly from the command line without arguments, relying on hardcoded paths to output CSVs and NetCDFs in the `results/` or `vectors/` directories. 

```bash
python generate_cross_index_plot.py
```