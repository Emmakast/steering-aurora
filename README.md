# Steering Aurora

This repository contains a modular framework for implementing **Contrastive Activation Addition (CAA) steering** on the Microsoft Aurora (0.25° Fine-Tuned) atmospheric foundation model. The primary objective is to physically steer large-scale meteorological phenomena (such as AO, ENSO, MJO, NAO, PNA, and AAO) by intervening in the model's latent space during inference.

## Project Structure

The project is broken down into several modular components, each with its own specialized scripts and dedicated `README.md` containing detailed usage instructions:

- **[`Data_loader/`](Data_loader/README.md)**  
  Scripts for downloading High-Resolution (HRES) T0 data from WeatherBench2 and Hugging Face, running Aurora predictions, and extracting internal latent representations and attention weights using PyTorch hooks.

- **[`Date_finder/`](Date_finder/README.md)**  
  A unified module to generate contrastive dataset splits. It automatically isolates "Active" and "Neutral" dates based on specific climate index thresholds while preventing cross-phenomenon contamination and matching seasonal distributions.

- **[`Index_calculator/`](Index_calculator/README.md)**  
  Scripts to compute Empirical Orthogonal Functions (EOFs) and calculate various atmospheric and oceanic indices (NAO, PNA, AAO, AO, ENSO, MJO) from NetCDF datasets and model forecasts to evaluate the physical impact of the steering vectors.

- **[`Prior_analysis/`](Prior_analysis/README.md)**  
  Tools for foundational analysis, including validating the local Aurora predictions against official WeatherBench2 baselines and evaluating precision errors when casting latents from FP32 to FP16.

- **[`Steering/`](Steering/README.md)**  
  The core steering implementation. Contains scripts to compute steering vectors from previously extracted latents and inject them back into the model dynamically during inference. Includes support for spatial masking, multi-layer injection, and ablation studies.

- **[`Visualization/`](Visualization/README.md)**  
  Python scripts utilizing `cartopy` and `matplotlib` to generate high-quality figures, maps, grid plots, and tables for evaluating and analyzing the outcomes of the steering experiments.

## Prerequisites

Each module has its own specific dependencies, but generally, this repository relies heavily on the Earth Science Python ecosystem and PyTorch. 

Core dependencies include:
- `torch`
- `xarray`
- `pandas`
- `numpy`
- `matplotlib`, `seaborn`, `cartopy` (for Visualization)
- `eofs`, `windspharm` (for Index Calculation)
- `aurora`, `huggingface_hub`, `fsspec` (for Data Loading & Model Execution)

## Getting Started

1. **Find Dates**: Start by using the `Date_finder` to generate contrastive (Active vs Neutral) splits for your target climate phenomenon.
2. **Download & Extract**: Use the `Data_loader` to download the initial states for those dates and extract the baseline latent representations from the Aurora model.
3. **Calculate Indices (Optional)**: If you are evaluating a new phenomenon, use the `Index_calculator` to generate the EOFs required for downstream evaluation.
4. **Steer**: Use the `Steering` module to compute the steering vector from the extracted latents and inject it during a new forecast rollout.
5. **Visualize**: Finally, run the `Visualization` scripts to analyze the causal difference between the base and steered forecasts.

For detailed command-line arguments and configuration options, please refer to the specific `README.md` inside each sub-directory.