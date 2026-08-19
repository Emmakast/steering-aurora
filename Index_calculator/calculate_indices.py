import argparse
import logging
import os
import xarray as xr
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_indices(target_file, climatology, eofs_dir):
    def std_coords(d):
        r = {}
        if 'latitude' in d.coords: r['latitude'] = 'lat'
        if 'longitude' in d.coords: r['longitude'] = 'lon'
        if 'valid_time' in d.coords: r['valid_time'] = 'time'
        if 'geopotential' in d.variables: r['geopotential'] = 'z'
        return d.rename(r) if r else d

    if isinstance(target_file, str):
        logging.info(f"Loading target file: {target_file}")
        target_ds = std_coords(xr.open_dataset(target_file))
        if 'time' not in target_ds.coords:
            import re, pandas as pd
            match = re.search(r'(\d{8})_(\d{4})', target_file)
            if match:
                from datetime import datetime
                init_time = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M")
                target_time = pd.to_datetime(init_time) + pd.Timedelta(hours=72)
                target_ds = target_ds.assign_coords(time=[target_time])
    else:
        target_ds = std_coords(target_file)
    
    if isinstance(climatology, str):
        logging.info(f"Loading climatology file: {climatology}")
        if climatology.startswith('gs://') or climatology.endswith('.zarr'):
            clim_ds = std_coords(xr.open_zarr(climatology, consolidated=True))
        else:
            clim_ds = std_coords(xr.open_dataset(climatology))
    else:
        clim_ds = std_coords(climatology)
    
    var_name = 'z'
    if var_name not in target_ds.data_vars:
        logging.warning(f"'{var_name}' not found. Available vars: {list(target_ds.data_vars.keys())}")
    
    indices = ['NAO', 'PNA', 'AAO', 'AO']
    results = {}
    
    for index in indices:
        eof_path = os.path.join(eofs_dir, f"{index.lower()}_loading_pattern.nc")
        if not os.path.exists(eof_path):
            logging.error(f"EOF file not found for {index}: {eof_path}")
            continue
            
        logging.info(f"Calculating {index}...")
        eof_ds = std_coords(xr.open_dataset(eof_path))
        eof_pattern = eof_ds['eof']
        pc_std = eof_ds['daily_pc_std'] if 'daily_pc_std' in eof_ds else eof_ds['pc_std']
        
        lat_min, lat_max = float(eof_pattern.lat.min()), float(eof_pattern.lat.max())
        lon_min, lon_max = float(eof_pattern.lon.min()), float(eof_pattern.lon.max())
        lon_format = '180' if eof_pattern.lon.min() < 0 else '360'
        
        def standardize_lon(ds, fmt):
            if fmt == '180':
                return ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
            else:
                return ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
        
        target_subset = standardize_lon(target_ds, lon_format)
        clim_subset = standardize_lon(clim_ds, lon_format)
        
        if index in ['NAO', 'PNA']:
            level_val = 500
        elif index == 'AAO':
            level_val = 700
        elif index == 'AO':
            level_val = 1000
        else:
            level_val = 500
            
        if 'level' in target_subset.coords:
            target_subset = target_subset.sel(level=level_val, method='nearest')
        if 'level' in clim_subset.coords:
            clim_subset = clim_subset.sel(level=level_val, method='nearest')
            
        def sel_domain(ds, lat_bnds, lon_bnds):
            lat_slice = slice(lat_bnds[1], lat_bnds[0]) if ds.lat.values[0] > ds.lat.values[-1] else slice(lat_bnds[0], lat_bnds[1])
            return ds.sel(lat=lat_slice, lon=slice(lon_bnds[0], lon_bnds[1]))
            
        target_sliced = sel_domain(target_subset, (lat_min, lat_max), (lon_min, lon_max))
        clim_sliced = sel_domain(clim_subset, (lat_min, lat_max), (lon_min, lon_max))
        
        target_var = target_sliced[var_name]
        clim_var = clim_sliced[var_name]
        
        if 'time' in target_var.coords:
            anom_list = []
            target_var_expanded = target_var if 'time' in target_var.dims else target_var.expand_dims('time')
            
            for t in target_var_expanded.time:
                t_val = t.values
                t_dt = xr.DataArray(t_val).dt
                
                c_var = clim_var
                if 'dayofyear' in c_var.coords:
                    c_var = c_var.sel(dayofyear=t_dt.dayofyear)
                if 'hour' in c_var.coords:
                    c_var = c_var.sel(hour=t_dt.hour)
                elif 'time' in c_var.coords and 'dayofyear' not in c_var.coords:
                    # fallback to match closest time if structured differently
                    pass 
                    
                anom_list.append(target_var_expanded.sel(time=t) - c_var)
                
            anom = xr.concat(anom_list, dim='time')
            if 'time' not in target_var.dims:
                anom = anom.squeeze('time')
        else:
            anom = target_var - clim_var
            
        # Apply identical weighting to the daily anomaly
        lat_weights = np.sqrt(np.clip(np.cos(np.deg2rad(anom.lat)), 0, None))
        anom_weighted = anom * lat_weights
        
        # Spatial dot product and standardization
        raw_projection = (anom_weighted * eof_pattern).sum(dim=['lat', 'lon'])
        index_value = raw_projection / pc_std
        
        # Print the final scalar index values
        logging.info(f"{index} Index Value:\n{index_value.values}")
        results[index] = index_value
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate standard atmospheric teleconnection indices (NAO, PNA, AAO, AO).')
    parser.add_argument('--target', type=str, required=True, help='Path to target NetCDF file (forecast/state)')
    parser.add_argument('--climatology', type=str, required=True, help='Path to climatology NetCDF/Zarr file')
    parser.add_argument('--eofs_dir', type=str, required=True, help='Directory containing the static EOF patterns')
    
    args = parser.parse_args()
    calculate_indices(args.target, args.climatology, args.eofs_dir)
