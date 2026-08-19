# Date Finder

This module contains a unified script to generate contrastive dataset splits (Active and Neutral sets) for various climate indices based on specific threshold and distributional rules. 

Instead of dealing with multiple fragmented extraction scripts, you can now generate isolated datasets using a single configuration.

## Features

- **Threshold Isolation**: Automatically isolates dates where the target climate index strongly deviates from the climatological mean (e.g., AO > 3.0, ENSO < -0.8).
- **Cross-Phenomena Isolation**: When picking active dates for a target phenomenon (like ENSO), the script automatically filters out dates where *any other overlapping phenomenon* (like MJO) is highly active. This prevents steering vectors from capturing mixed signals.
- **Monthly Distributional Matching**: For every active date selected, the script samples a corresponding "neutral" date (where the target index is closest to 0) from the *exact same calendar month*. This guarantees that your contrastive pairs have identical seasonal distributions, preventing your model from accidentally learning background summer/winter shifts.

## Prerequisites

Ensure you have your raw index `.csv` and `.txt` files in a centralized data directory. 

Required files in your data directory (names are partially matched):
- `norm.daily.ao.cdas.z1000.*.csv`
- `norm.daily.aao.cdas.z700.*.csv`
- `norm.daily.nao.cdas.z500.*.csv`
- `norm.daily.pna.cdas.z500.*.csv`
- `rmm.74toRealtime.txt` (MJO)
- `soi.long.csv` (ENSO)

## Usage

You can generate the target dates for a specific phenomenon, or for all of them at once.

```bash
python find_contrastive_dates.py \
    --data_dir /path/to/your/data/directory \
    --target ENSO \
    --output target_dates_enso.csv
```

### Supported Targets
- `AO`
- `AAO`
- `MJO`
- `NAO`
- `PNA`
- `ENSO`
- `ALL` (Generates rows for all phenomena combined into one CSV)

## Output Format
The resulting CSV will contain the following columns:
- `Year`
- `Month`
- `Day`
- `Phenomenon` (e.g., ENSO)
- `Type` (Active or Neutral)
- `Value` (The actual index value on that day)
