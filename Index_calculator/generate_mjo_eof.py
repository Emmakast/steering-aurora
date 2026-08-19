import argparse
import logging
import os
import xarray as xr
import numpy as np
from eofs.xarray import Eof
from windspharm.xarray import VectorWind

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

def slice_tropics(ds):
    """Slice to 15°N-15°S and ensure longitudes are 0-360 continuously."""
    # Standardize longitude to 0-360 for consistency in MJO EOF vector
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    
    # Domain: 15°N to 15°S
    lat_min, lat_max = -15, 15
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    else:
        ds = ds.sel(lat=slice(lat_min, lat_max))
    return ds

def extract_variables(ds):
    """Attempt to extract U850, U200, and V200 from dataset."""
    vars_dict = {}
    
    # U850 and U200
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
        
    if len(vars_dict) < 3:
        logging.warning(f"Could not find all required variables. Found: {list(vars_dict.keys())}")
        
    return vars_dict

def process_chunk(args):
    start, chunk_size, n_time, input_zarr = args
    # We re-open the zarr store in the worker to avoid pickling the global dataset or locks
    import xarray as xr
    from windspharm.xarray import VectorWind
    ds_local = xr.open_zarr(input_zarr)
    ds_local = standardize_coords(ds_local)
    
    ds_local = ds_local.sel(time=slice('1990-01-01', '2019-12-31'))
    
    vars_local = extract_variables(ds_local)
    u200_local = vars_local['u200']
    v200_local = vars_local['v200']
    u850_local = vars_local['u850']
    
    end = min(start + chunk_size, n_time)
    u_chunk = u200_local.isel(time=slice(start, end)).load()
    v_chunk = v200_local.isel(time=slice(start, end)).load()
    u850_chunk = u850_local.isel(time=slice(start, end)).load()
    w = VectorWind(u_chunk, v_chunk)
    vp_chunk = w.velocitypotential()
    
    def process_wind_component(da):
        lat_min, lat_max = -15, 15
        if da.lat.values[0] > da.lat.values[-1]:
            da_tropics = da.sel(lat=slice(lat_max, lat_min))
        else:
            da_tropics = da.sel(lat=slice(lat_min, lat_max))
        da_1d = da_tropics.mean(dim='lat')
        da_1d = da_1d.assign_coords(lon=(da_1d.lon % 360)).sortby('lon')
        return da_1d

    vp_ds = xr.Dataset({'vp200': vp_chunk})
    vp_tropics = slice_tropics(vp_ds)['vp200']
    vp_1d = vp_tropics.mean(dim='lat').compute()
    
    u850_1d = process_wind_component(u850_chunk).compute()
    u200_1d = process_wind_component(u_chunk).compute()
    
    return start, vp_1d, u850_1d, u200_1d

def generate_mjo_eofs(input_zarr, output_dir):
    logging.info(f"Loading Zarr store from {input_zarr}")
    ds = xr.open_zarr(input_zarr)
    ds = standardize_coords(ds)
    
    logging.info("Slicing time (1990-2019)...")
    ds = ds.sel(time=slice('1990-01-01', '2019-12-31'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    vars_dict_global = extract_variables(ds)
    u200 = vars_dict_global['u200']
    v200 = vars_dict_global['v200']
    u850 = vars_dict_global['u850']

    logging.info("Calculating velocity potential globally in chunks and reducing...")
    chunk_size = 30  # 30 day chunks to avoid OverflowError in windspharm Fortran code
    n_time = len(u200.time)

    import concurrent.futures
    vp200_1d_dict = {}
    u850_1d_dict = {}
    u200_1d_dict = {}
    # Use ProcessPoolExecutor to be 100% safe regarding Fortran thread-safety and xarray memory leaks
    # Use 16 workers to fully utilize the 1/8 node Snellius allocation
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        args_list = [(start, chunk_size, n_time, input_zarr) for start in range(0, n_time, chunk_size)]
        futures = {executor.submit(process_chunk, arg): arg[0] for arg in args_list}
        for future in concurrent.futures.as_completed(futures):
            try:
                start, vp_1d, u850_1d_chk, u200_1d_chk = future.result()
                vp200_1d_dict[start] = vp_1d
                u850_1d_dict[start] = u850_1d_chk
                u200_1d_dict[start] = u200_1d_chk
                logging.info(f"Completed chunk starting at {start}/{n_time}")
            except Exception as e:
                logging.error(f"Error processing chunk starting at {futures[future]}: {e}")
                raise e
                
    vp200_1d_chunks = [vp200_1d_dict[start] for start in sorted(vp200_1d_dict.keys())]
    vp200_1d = xr.concat(vp200_1d_chunks, dim='time')
    
    u850_1d_chunks = [u850_1d_dict[start] for start in sorted(u850_1d_dict.keys())]
    u850_1d = xr.concat(u850_1d_chunks, dim='time')
    
    u200_1d_chunks = [u200_1d_dict[start] for start in sorted(u200_1d_dict.keys())]
    u200_1d = xr.concat(u200_1d_chunks, dim='time')
    
    logging.info("Calculating anomalies...")
    def get_anomalies(da):
        clim = da.groupby('time.dayofyear').mean('time')
        return da.groupby('time.dayofyear') - clim
        
    vp200_anom = get_anomalies(vp200_1d)
    u850_anom = get_anomalies(u850_1d)
    u200_anom = get_anomalies(u200_1d)
    
    logging.info("Normalizing by global standard deviations...")
    vp200_std = vp200_anom.std(dim=['time', 'lon'])
    u850_std = u850_anom.std(dim=['time', 'lon'])
    u200_std = u200_anom.std(dim=['time', 'lon'])
    
    vp200_norm = vp200_anom / vp200_std
    u850_norm = u850_anom / u850_std
    u200_norm = u200_anom / u200_std
    
    logging.info("Concatenating into combined vector...")
    n_lon = len(vp200_norm.lon)
    vp200_c = vp200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon)).drop_vars('level', errors='ignore')
    u850_c = u850_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + n_lon).drop_vars('level', errors='ignore')
    u200_c = u200_norm.rename({'lon': 'combined_lon'}).assign_coords(combined_lon=np.arange(n_lon) + 2*n_lon).drop_vars('level', errors='ignore')
    
    combined = xr.concat([vp200_c, u850_c, u200_c], dim='combined_lon')
    combined = combined.compute()
    
    logging.info("Running EOF solver...")
    solver = Eof(combined)
    
    eofs = solver.eofs(neofs=2).squeeze()
    
    eof1 = eofs.sel(mode=0).drop_vars('mode')
    eof2 = eofs.sel(mode=1).drop_vars('mode')
    
    logging.info("Saving results...")
    
    # Calculate PC standard deviations from eigenvalues
    eigenvalues = solver.eigenvalues(neigs=2)
    pc1_std = np.sqrt(eigenvalues.sel(mode=0).drop_vars('mode'))
    pc2_std = np.sqrt(eigenvalues.sel(mode=1).drop_vars('mode'))
    
    out_ds = xr.Dataset({
        'eof1': eof1,
        'eof2': eof2,
        'vp200_std': vp200_std,
        'u850_std': u850_std,
        'u200_std': u200_std,
        'pc1_std': pc1_std,
        'pc2_std': pc2_std
    })
    
    out_path = os.path.join(output_dir, "mjo_loading_pattern.nc")
    out_ds.to_netcdf(out_path)
    logging.info(f"Saved MJO EOFs to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Combined EOF loading patterns for MJO (RMM).')
    parser.add_argument('--input', type=str, required=True, help='Path to input Zarr store (e.g., ERA5)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated NetCDF files')
    
    args = parser.parse_args()
    generate_mjo_eofs(args.input, args.output_dir)
