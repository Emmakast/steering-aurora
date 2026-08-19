import argparse
import logging
import os
import xarray as xr
import numpy as np
from eofs.xarray import Eof

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def slice_domain(ds, lat_range, lon_range, lon_format='180'):
    """Slice dataset optimally depending on the region to avoid discontinuous longitude arrays."""
    # Standardize longitude
    if lon_format == '180':
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    elif lon_format == '360':
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    
    # Slice latitude based on array ordering
    lat_min, lat_max = sorted(lat_range)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))
        
    # Slice longitude
    lon_min, lon_max = sorted(lon_range)
    ds = ds.sel(lon=slice(lon_min, lon_max))
    
    return ds

INDICES = {
    'NAO': {
        'level': 500,
        'lat_range': (20, 90),
        'lon_range': (-90, 40),
        'lon_format': '180',
        'check_lat': 65,
        'check_lon': -20,
        'expected_sign': -1
    },
    'PNA': {
        'level': 500,
        'lat_range': (20, 85),
        'lon_range': (160, 300),  # 160E to 60W (360 - 60 = 300)
        'lon_format': '360',
        'check_lat': 50,
        'check_lon': 200,  # 160W = 200E
        'expected_sign': -1
    },
    'AAO': {
        'level': 700,
        'lat_range': (-90, -20),
        'lon_range': (0, 360),
        'lon_format': '360',
        'check_lat': -90,
        'check_lon': 0,
        'expected_sign': -1
    },
    'AO': {
        'level': 1000,
        'lat_range': (20, 90),
        'lon_range': (0, 360),
        'lon_format': '360',
        'check_lat': 90,
        'check_lon': 0,
        'expected_sign': -1
    }
}

def generate_single_eof(input_zarr, output_dir, name):
    """Generate the EOF loading pattern for a single atmospheric index."""
    params = INDICES[name]
    
    logging.info(f"Loading Zarr store from {input_zarr}")
    
    # ── Key optimisation: select only the variable and level we need ────────
    # This avoids loading all variables across all levels from the remote store.
    var_name = 'z'
    ds = xr.open_zarr(input_zarr)
    
    def std_coords(d):
        r = {}
        if 'latitude' in d.coords: r['latitude'] = 'lat'
        if 'longitude' in d.coords: r['longitude'] = 'lon'
        if 'valid_time' in d.coords: r['valid_time'] = 'time'
        if 'geopotential' in d.variables: r['geopotential'] = 'z'
        return d.rename(r) if r else d

    ds = std_coords(ds)
    
    if var_name not in ds.data_vars:
        logging.warning(f"'{var_name}' not found in data vars. Available vars: {list(ds.data_vars.keys())}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Processing {name}...")
    
    # Select *only* the target variable to avoid touching other vars in the store
    da = ds[var_name]
    
    # Select pressure level early – drops entire dimension before any I/O
    if 'level' in da.coords:
        da = da.sel(level=params['level'], method='nearest')
    
    # Slice to the geographic domain early – massively reduces data volume
    # We build a tiny single-variable dataset so slice_domain works unchanged
    ds_subset = da.to_dataset(name=var_name)
    ds_sliced = slice_domain(ds_subset, params['lat_range'], params['lon_range'], params['lon_format'])
    da_sliced = ds_sliced[var_name]
    
    # Resample to monthly means ('1MS') to isolate low-frequency modes
    logging.info(f"  Resampling {name} to monthly means...")
    da_monthly = da_sliced.resample(time='1MS').mean()
    
    # Calculate monthly anomalies
    logging.info(f"  Calculating anomalies for {name}...")
    climatology = da_monthly.groupby('time.month').mean('time')
    anomalies = (da_monthly.groupby('time.month') - climatology).compute()
    
    # Area Weighting
    # You MUST weight the data before feeding it to the eofs solver
    logging.info(f"  Applying area weighting for {name}...")
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(anomalies.lat)), 0, None))
    anomalies_weighted = anomalies * weights
    
    logging.info(f"  Computing EOF for {name}...")
    solver = Eof(anomalies_weighted)
    
    # Retrieve EOF1 and PC1
    eof1 = solver.eofs(neofs=1).squeeze()
    pc1 = solver.pcs(npcs=1).squeeze()
    
    # Extract the standard deviation of PC1
    pc_std = pc1.std(dim='time')
    
    # Check polarity
    check_point = eof1.sel(lat=params['check_lat'], lon=params['check_lon'], method='nearest')
    if np.sign(check_point.values) != np.sign(params['expected_sign']):
        logging.info(f"  Reversing polarity for {name}...")
        eof1 = eof1 * -1
        
    # Create output dataset
    out_ds = xr.Dataset(
        {
            'eof': eof1,
            'pc_std': pc_std
        }
    )
    
    out_path = os.path.join(output_dir, f"{name.lower()}_loading_pattern.nc")
    out_ds.to_netcdf(out_path)
    logging.info(f"  Saved {name} to {out_path}")


def generate_eofs(input_zarr, output_dir, index_name=None):
    """Generate EOF loading patterns. If index_name is given, only that index is computed."""
    if index_name:
        names = [index_name.upper()]
    else:
        names = list(INDICES.keys())
    
    for name in names:
        if name not in INDICES:
            logging.error(f"Unknown index '{name}'. Choose from: {list(INDICES.keys())}")
            continue
        generate_single_eof(input_zarr, output_dir, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EOF loading patterns for atmospheric indices.')
    parser.add_argument('--input', type=str, required=True, help='Path to input Zarr store (e.g., ERA5)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated NetCDF files')
    parser.add_argument('--index', type=str, choices=['NAO', 'PNA', 'AAO', 'AO'], help='Specific index to generate (optional, generates all if omitted)')
    
    args = parser.parse_args()
    generate_eofs(args.input, args.output_dir, args.index)
