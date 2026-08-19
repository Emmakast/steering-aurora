import argparse
import logging
import os
import xarray as xr
import numpy as np
import pandas as pd
from windspharm.xarray import VectorWind

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if 'valid_time' in ds.coords:
        rename_dict['valid_time'] = 'time'
    return ds.rename(rename_dict) if rename_dict else ds

def slice_tropics(ds):
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    lat_min, lat_max = -15, 15
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))
    return ds

def extract_variables(ds):
    vars_dict = {}
    
    if 'u850' in ds.data_vars:
        vars_dict['u850'] = ds['u850']
    elif 'u' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u850'] = ds['u'].sel(level=850, method='nearest')
    elif 'u_component_of_wind' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u850'] = ds['u_component_of_wind'].sel(level=850, method='nearest')
        
    if 'u200' in ds.data_vars:
        vars_dict['u200'] = ds['u200']
    elif 'u' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u200'] = ds['u'].sel(level=200, method='nearest')
    elif 'u_component_of_wind' in ds.data_vars and 'level' in ds.coords:
        vars_dict['u200'] = ds['u_component_of_wind'].sel(level=200, method='nearest')

    if 'v200' in ds.data_vars:
        vars_dict['v200'] = ds['v200']
    elif 'v' in ds.data_vars and 'level' in ds.coords:
        vars_dict['v200'] = ds['v'].sel(level=200, method='nearest')
    elif 'v_component_of_wind' in ds.data_vars and 'level' in ds.coords:
        vars_dict['v200'] = ds['v_component_of_wind'].sel(level=200, method='nearest')
        
    return vars_dict

def get_anomaly(t_var, c_var):
    if 'time' in t_var.coords:
        anom_list = []
        t_var_expanded = t_var if 'time' in t_var.dims else t_var.expand_dims('time')
        
        for t in t_var_expanded.time:
            t_val = t.values
            t_dt = pd.to_datetime(t_val)
            
            c_slice = c_var
            if 'dayofyear' in c_slice.coords:
                c_slice = c_slice.sel(dayofyear=t_dt.dayofyear)
            elif 'month' in c_slice.coords:
                c_slice = c_slice.sel(month=t_dt.month)
            if 'hour' in c_slice.coords:
                c_slice = c_slice.sel(hour=t_dt.hour)
            
            anom_list.append(t_var_expanded.sel(time=t) - c_slice)
            
        anom = xr.concat(anom_list, dim='time')
        if 'time' not in t_var.dims:
            anom = anom.squeeze('time')
        return anom
    else:
        return t_var - c_var

def calculate_mjo(target_file, climatology, eof_file):
    if isinstance(target_file, str):
        target_ds = standardize_coords(xr.open_dataset(target_file))
        if 'time' not in target_ds.coords:
            import re
            match = re.search(r'(\d{8})_(\d{4})', target_file)
            if match:
                from datetime import datetime
                init_time = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M")
                target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
                target_ds = target_ds.assign_coords(time=[target_time])
    else:
        target_ds = standardize_coords(target_file)
    
    if isinstance(climatology, str):
        if climatology.startswith('gs://') or climatology.endswith('.zarr'):
            clim_ds = standardize_coords(xr.open_zarr(climatology, consolidated=True))
        else:
            clim_ds = standardize_coords(xr.open_dataset(climatology))
    else:
        clim_ds = climatology
    
    if isinstance(eof_file, str):
        eof_ds = xr.open_dataset(eof_file)
    else:
        eof_ds = eof_file
    
    import time
    t0 = time.time()
    
    target_vars_g = extract_variables(target_ds)
    clim_vars_g = extract_variables(clim_ds)
    print(f"Extracted vars: {time.time() - t0:.2f}s")
    
    # 1. Compute anomalies globally FIRST to avoid computing VectorWind on the huge climatology
    t1 = time.time()
    u200_anom = get_anomaly(target_vars_g['u200'], clim_vars_g['u200'])
    v200_anom = get_anomaly(target_vars_g['v200'], clim_vars_g['v200'])
    u850_anom = get_anomaly(target_vars_g['u850'], clim_vars_g['u850'])
    print(f"Calculated anomaly graphs: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    # Force evaluation here to see if data loading is the bottleneck
    u200_anom = u200_anom.compute()
    v200_anom = v200_anom.compute()
    u850_anom = u850_anom.compute()
    print(f"Computed anomalies (loading data): {time.time() - t2:.2f}s")
    
    # 2. Ensure global grid for windspharm by interpolating to 721 latitudes if needed
    t3 = time.time()
    if len(u200_anom.lat) == 720:
        new_lat = np.linspace(90, -90, 721)
        u200_anom_w = u200_anom.interp(lat=new_lat, kwargs={'fill_value': 'extrapolate'})
        v200_anom_w = v200_anom.interp(lat=new_lat, kwargs={'fill_value': 'extrapolate'})
    else:
        u200_anom_w = u200_anom
        v200_anom_w = v200_anom
    print(f"Interpolation: {time.time() - t3:.2f}s")

    # 3. Compute velocity potential of the anomaly
    t4 = time.time()
    w = VectorWind(u200_anom_w, v200_anom_w)
    print(f"Initialized VectorWind: {time.time() - t4:.2f}s")
    
    t5 = time.time()
    vp200_anom = w.velocitypotential()
    print(f"Computed velocitypotential: {time.time() - t5:.2f}s")
    
    # Slice to tropics
    vp200_tropics = slice_tropics(vp200_anom)
    u850_tropics = slice_tropics(u850_anom)
    u200_tropics = slice_tropics(u200_anom)
    
    combined_list = []
    
    for var_key, anom in zip(['vp200', 'u850', 'u200'], [vp200_tropics, u850_tropics, u200_tropics]):
        # Meridionally average (15N-15S)
        anom_1d = anom.mean(dim='lat')
        
        # Normalize with saved standard deviations
        std_val = eof_ds[f"{var_key}_std"]
        anom_norm = anom_1d / std_val
        combined_list.append(anom_norm)
        
    # Concatenate and project
    vp200_norm, u850_norm, u200_norm = combined_list
    n_lon = len(vp200_norm.lon)
    
    vp200_c = vp200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon)).drop_vars('level', errors='ignore')
    u850_c = u850_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + n_lon).drop_vars('level', errors='ignore')
    u200_c = u200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + 2*n_lon).drop_vars('level', errors='ignore')
    
    combined = xr.concat([vp200_c, u850_c, u200_c], dim='combined_lon')
    
    eof1 = eof_ds['eof1']
    eof2 = eof_ds['eof2']
    
    rmm1 = (combined * eof1).sum(dim='combined_lon')
    rmm2 = (combined * eof2).sum(dim='combined_lon')
    
    # Normalize by PC standard deviation to get standardized RMM index
    # Prefer daily_pc_std (computed from daily projections) over monthly pc_std
    if 'daily_pc1_std' in eof_ds and 'daily_pc2_std' in eof_ds:
        rmm1 = rmm1 / eof_ds['daily_pc1_std']
        rmm2 = rmm2 / eof_ds['daily_pc2_std']
    elif 'pc1_std' in eof_ds and 'pc2_std' in eof_ds:
        logging.warning("MJO EOF file does not contain daily_pc1_std/daily_pc2_std. Falling back to monthly pc_std. Run compute_daily_pc_std.py to fix.")
        rmm1 = rmm1 / eof_ds['pc1_std']
        rmm2 = rmm2 / eof_ds['pc2_std']
    else:
        logging.warning("MJO EOF file does not contain pc1_std and pc2_std. Using approximate scaling (22.75) to prevent RMM inflation. Please re-run generate_mjo_eof.py.")
        rmm1 = rmm1 / 22.75
        rmm2 = rmm2 / 22.75
    
    angle = np.arctan2(rmm2, rmm1) * 180.0 / np.pi
    angle = (angle + 360) % 360
    mjo_phase = np.floor(((angle + 22.5) % 360) / 45.0) + 1
    mjo_amp = np.sqrt(rmm1**2 + rmm2**2)
    
    return {
        'rmm1': rmm1,
        'rmm2': rmm2,
        'phase': mjo_phase,
        'amplitude': mjo_amp
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MJO RMM Indices.')
    parser.add_argument('--target', type=str, required=True, help='Path to target NetCDF file (forecast/state)')
    parser.add_argument('--climatology', type=str, required=True, help='Path to climatology NetCDF file')
    parser.add_argument('--eof_file', type=str, required=True, help='Path to static MJO EOF NetCDF file')
    
    args = parser.parse_args()
    calculate_mjo(args.target, args.climatology, args.eof_file)
