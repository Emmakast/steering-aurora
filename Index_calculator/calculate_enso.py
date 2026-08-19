import argparse
import logging
import xarray as xr
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize latitude and longitude coordinate names."""
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords:
        rename_dict['valid_time'] = 'time'
    return ds.rename(rename_dict) if rename_dict else ds

def slice_nino34(ds, lat_name='lat', lon_name='lon'):
    """Slice dataset to Niño 3.4 region, handling different longitude formats gracefully."""
    # Latitude: 5°N to 5°S
    lat_min, lat_max = -5, 5
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.sel({lat_name: slice(lat_max, lat_min)})
    else:
        ds = ds.sel({lat_name: slice(lat_min, lat_max)})
        
    # Standardize longitude to 0-360 to simplify the slice
    ds = ds.assign_coords({lon_name: (ds[lon_name] % 360)}).sortby(lon_name)
    
    # Niño 3.4 is 120°W to 170°W -> 190°E to 240°E
    ds = ds.sel({lon_name: slice(190, 240)})
    return ds

def calculate_enso(target_file, climatology_file):
    logging.info(f"Loading target file: {target_file}")
    target_ds = standardize_coords(xr.open_dataset(target_file))
    
    logging.info(f"Loading climatology file: {climatology_file}")
    if climatology_file.startswith('gs://') or climatology_file.endswith('.zarr'):
        clim_ds = standardize_coords(xr.open_zarr(climatology_file, consolidated=True))
    else:
        clim_ds = standardize_coords(xr.open_dataset(climatology_file))
    
    # Identify SST variable (with 2t fallback for Aurora)
    var_name = None
    clim_var_name = None
    for v in ['sst', 'SST', 'tos', 'sea_surface_temperature']:
        if v in target_ds.data_vars:
            var_name = v
            clim_var_name = v
            break
            
    if var_name is None and '2t' in target_ds.data_vars:
        logging.warning("SST not found. Using '2t' (2-meter temperature) as proxy.")
        var_name = '2t'
        clim_var_name = '2m_temperature'
            
    if var_name is None:
        raise ValueError(f"SST variable not found. Available vars: {list(target_ds.data_vars.keys())}")
        
    target_sliced = slice_nino34(target_ds)
    clim_sliced = slice_nino34(clim_ds)
    
    target_var = target_sliced[var_name]
    clim_var = clim_sliced[clim_var_name]
    
    # 1. Calculate anomalies
    logging.info("Calculating SST anomalies...")
    if 'time' in target_var.coords:
        anom_list = []
        target_var_expanded = target_var if 'time' in target_var.dims else target_var.expand_dims('time')
        
        for t in target_var_expanded.time:
            t_val = t.values
            t_dt = xr.DataArray(t_val).dt
            
            c_var = clim_var
            if 'month' in c_var.coords and 'time' not in c_var.coords:
                c_var = c_var.sel(month=t_dt.month)
            elif 'dayofyear' in c_var.coords:
                c_var = c_var.sel(dayofyear=t_dt.dayofyear)
            elif 'hour' in c_var.coords:
                c_var = c_var.sel(hour=t_dt.hour)
                
            anom_list.append(target_var_expanded.sel(time=t) - c_var)
            
        anom = xr.concat(anom_list, dim='time')
        if 'time' not in target_var.dims:
            anom = anom.squeeze('time')
    else:
        anom = target_var - clim_var
        
    # 2. Area-weighted spatial mean
    logging.info("Calculating area-weighted spatial mean over Niño 3.4 region...")
    weights = np.cos(np.deg2rad(anom.lat))
    weights.name = 'weights'
    anom_weighted_mean = anom.weighted(weights).mean(dim=['lat', 'lon'])
    
    # 3. Apply a 3-month rolling mean to the resulting time series
    logging.info("Applying 3-month rolling mean (assuming monthly input data)...")
    if 'time' in anom_weighted_mean.dims and len(anom_weighted_mean.time) >= 3:
        # Standard ONI uses a 3-month rolling mean.
        oni = anom_weighted_mean.rolling(time=3, center=True).mean()
    else:
        logging.warning("Time series is shorter than 3 steps or has no time dimension. Skipping rolling mean.")
        oni = anom_weighted_mean
        
    logging.info(f"ONI Index Values: \n{oni.values}")
    return oni

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Oceanic Niño Index (ONI).')
    parser.add_argument('--target', type=str, required=True, help='Path to target NetCDF file (forecast/state)')
    parser.add_argument('--climatology', type=str, required=True, help='Path to climatology NetCDF file')
    
    args = parser.parse_args()
    calculate_enso(args.target, args.climatology)
