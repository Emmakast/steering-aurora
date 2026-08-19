import os
import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

THRESHOLDS = {
    'AO': {'metric': 'ao_index', 'threshold': 3.0, 'op': 'gt', 'neutral_target': 0.0},
    'AAO': {'metric': 'aao_index', 'threshold': 3.0, 'op': 'gt', 'neutral_target': 0.0},
    'MJO': {'metric': 'amplitude', 'threshold': 3.0, 'op': 'gt', 'neutral_target': 0.0},
    'NAO': {'metric': 'nao_index', 'threshold': 1.5, 'op': 'gt', 'neutral_target': 0.0},
    'PNA': {'metric': 'pna_index', 'threshold': 1.5, 'op': 'gt', 'neutral_target': 0.0},
    'ENSO': {'metric': 'soi_index', 'threshold': -0.8, 'op': 'lt', 'neutral_target': 0.0}
}

def is_active(row, phenomenon):
    config = THRESHOLDS[phenomenon]
    val = row.get(config['metric'])
    if pd.isna(val):
        return False
    if config['op'] == 'gt':
        return val > config['threshold']
    elif config['op'] == 'lt':
        return val < config['threshold']
    return False

def load_data(data_dir):
    """Loads and merges all indices into a single daily dataframe."""
    print("Loading data...")
    
    # MJO
    mjo_path = os.path.join(data_dir, 'rmm.74toRealtime.txt')
    mjo_cols = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'origin']
    mjo_df = pd.read_csv(mjo_path, skiprows=2, sep=r'\s+', names=mjo_cols, usecols=[0, 1, 2, 3, 4, 5, 6, 7], engine='python')
    mjo_df = mjo_df[['year', 'month', 'day', 'amplitude']]
    mjo_df['amplitude'] = pd.to_numeric(mjo_df['amplitude'], errors='coerce')
    mjo_df = mjo_df[(mjo_df['amplitude'] < 999) & (mjo_df['amplitude'] < 1e35)]
    
    # AO
    ao_df = pd.read_csv(os.path.join(data_dir, 'norm.daily.ao.cdas.z1000.19500101_current.csv'))
    ao_df.rename(columns={'ao_index_cdas': 'ao_index'}, inplace=True)
    
    # AAO
    aao_df = pd.read_csv(os.path.join(data_dir, 'norm.daily.aao.cdas.z700.19790101_current.csv'))
    aao_df.rename(columns={'aao_index_cdas': 'aao_index'}, inplace=True)
    
    # NAO
    # Note: assuming naming matches the typical CPC standard formatting
    nao_file = [f for f in os.listdir(data_dir) if 'nao.cdas' in f][0]
    nao_df = pd.read_csv(os.path.join(data_dir, nao_file))
    nao_col = [c for c in nao_df.columns if 'nao' in c.lower()][0]
    nao_df.rename(columns={nao_col: 'nao_index'}, inplace=True)
    
    # PNA
    pna_file = [f for f in os.listdir(data_dir) if 'pna.cdas' in f][0]
    pna_df = pd.read_csv(os.path.join(data_dir, pna_file))
    pna_col = [c for c in pna_df.columns if 'pna' in c.lower()][0]
    pna_df.rename(columns={pna_col: 'pna_index'}, inplace=True)
    
    # ENSO (SOI is monthly, so we merge it to daily by year/month)
    enso_df = pd.read_csv(os.path.join(data_dir, 'soi.long.csv'))
    # standardize Date parsing for SOI
    if 'Date' in enso_df.columns:
        enso_df['Date'] = pd.to_datetime(enso_df['Date'])
        enso_df['year'] = enso_df['Date'].dt.year
        enso_df['month'] = enso_df['Date'].dt.month
    enso_col = [c for c in enso_df.columns if 'SOI' in c.upper()][0]
    enso_df.rename(columns={enso_col: 'soi_index'}, inplace=True)
    enso_df = enso_df[enso_df['soi_index'] != -99.99][['year', 'month', 'soi_index']]
    
    # Merge all DataFrames on year, month, day
    df = ao_df[['year', 'month', 'day', 'ao_index']].merge(aao_df[['year', 'month', 'day', 'aao_index']], on=['year', 'month', 'day'], how='inner')
    df = df.merge(nao_df[['year', 'month', 'day', 'nao_index']], on=['year', 'month', 'day'], how='inner')
    df = df.merge(pna_df[['year', 'month', 'day', 'pna_index']], on=['year', 'month', 'day'], how='inner')
    df = df.merge(mjo_df, on=['year', 'month', 'day'], how='inner')
    
    # Merge SOI on year, month
    df = df.merge(enso_df, on=['year', 'month'], how='left')
    
    # Optional: Filter for a valid overlapping period (e.g., 1979 - 2022)
    df = df[(df['year'] >= 1979) & (df['year'] <= 2022)].dropna().reset_index(drop=True)
    return df

def find_contrastive_dates(df, target_phenomenon):
    print(f"Finding contrastive pairs for {target_phenomenon}...")
    
    config = THRESHOLDS[target_phenomenon]
    metric = config['metric']
    
    # 1. Identify all active dates
    # We use boolean indexing for speed
    if config['op'] == 'gt':
        active_mask = df[metric] > config['threshold']
    else:
        active_mask = df[metric] < config['threshold']
        
    active_df = df[active_mask].copy()
    
    # 2. Cross-phenomenon isolation
    # Drop dates from the active set if any OTHER phenomenon is also highly active
    for other_phenom in THRESHOLDS.keys():
        if other_phenom != target_phenomenon:
            other_config = THRESHOLDS[other_phenom]
            other_metric = other_config['metric']
            if other_config['op'] == 'gt':
                isolation_mask = active_df[other_metric] <= other_config['threshold']
            else:
                isolation_mask = active_df[other_metric] >= other_config['threshold']
            active_df = active_df[isolation_mask]
            
    print(f"Found {len(active_df)} active dates for {target_phenomenon} after cross-phenomenon isolation.")
    
    if len(active_df) == 0:
        print("Warning: No active dates found after isolation.")
        return pd.DataFrame()
        
    # 3. Monthly Distributional Matching
    # Build the neutral pool: sorted by absolute distance to neutral_target
    neutral_pool = df.copy()
    neutral_pool['abs_dist'] = (neutral_pool[metric] - config['neutral_target']).abs()
    neutral_pool = neutral_pool.sort_values(by='abs_dist')
    
    # Exclude dates that are already active
    active_indices = active_df.index
    neutral_pool = neutral_pool.drop(index=active_indices)
    
    selected_neutral_indices = set()
    results = []
    
    for _, active_row in active_df.iterrows():
        # Record active date
        results.append({
            'Year': int(active_row['year']),
            'Month': int(active_row['month']),
            'Day': int(active_row['day']),
            'Phenomenon': target_phenomenon,
            'Type': 'Active',
            'Value': active_row[metric]
        })
        
        # Find neutral matching month
        target_month = active_row['month']
        
        # Filter neutral pool for same month, and not already selected
        month_cands = neutral_pool[neutral_pool['month'] == target_month]
        
        # Pick the best one (closest to 0) that isn't selected
        found_neutral = False
        for idx, neut_row in month_cands.iterrows():
            if idx not in selected_neutral_indices:
                selected_neutral_indices.add(idx)
                results.append({
                    'Year': int(neut_row['year']),
                    'Month': int(neut_row['month']),
                    'Day': int(neut_row['day']),
                    'Phenomenon': target_phenomenon,
                    'Type': 'Neutral',
                    'Value': neut_row[metric]
                })
                found_neutral = True
                break
                
        if not found_neutral:
            print(f"Warning: Could not find a unique neutral match for month {target_month}.")
            
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Generate Contrastive Active and Neutral Dates.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing index CSVs")
    parser.add_argument('--target', type=str, required=True, choices=list(THRESHOLDS.keys()) + ['ALL'], help="Target phenomenon")
    parser.add_argument('--output', type=str, default='target_dates.csv', help="Output CSV path")
    
    args = parser.parse_args()
    
    df = load_data(args.data_dir)
    
    all_results = []
    
    targets = list(THRESHOLDS.keys()) if args.target == 'ALL' else [args.target]
    
    for t in targets:
        res_df = find_contrastive_dates(df, t)
        if not res_df.empty:
            all_results.append(res_df)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        print(f"Successfully wrote {len(final_df)} dates to {args.output}")
    else:
        print("No valid dates found to write.")

if __name__ == '__main__':
    main()
