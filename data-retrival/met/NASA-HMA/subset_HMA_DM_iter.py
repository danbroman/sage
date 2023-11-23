import sys, os, glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray

''' 
    High Mountain Asia 1 km 6-hourly Downscaled Meteorological Data 2003 to 2018, Version 1:
    https://nsidc.org/data/hma_dm_6h/versions/1
    https://doi.org/10.5067/CRN0E7YPPFGY

    Requried package installations: conda install -c conda-forge numpy pandas xarray dask netCDF4 rioxarray


'''
def subset(year, month, bounds, resolution, ncfile):
    # NASA HMA has an integer time dimension for each day (e.g., hours since 2003-01-01, 00:00 UTC, hours since 2003-01-02, 00:00 UTC, and ...)
    def modify_time_dim(ds):
        time_unit = pd.Timedelta('1' + ds['time'].long_name.split('since')[0].strip())
        time_day = pd.Timestamp(ds['time'].long_name.split('since')[1].split(',')[0].strip())
        ds['time'] = time_day + time_unit * ds['time']
        return ds

    path_files = os.path.join('raw', 'HMA_DM_6H_v01_{}{:02d}*.nc4'.format(year, month))
    files = glob.glob(path_files); files.sort()
    ds_allc = xr.open_mfdataset(files, concat_dim = 'time', combine = 'nested', decode_coords = 'all', preprocess = modify_time_dim, parallel = True)
    ds_allc = ds_allc.rio.write_crs(ds_allc['crs'].spatial_ref).rio.set_spatial_dims(x_dim = 'x', y_dim = 'y') # '+proj=lcc +lat_1=30 +lat_2=62 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '

    # reproject without remapping
    ds_4326_subset = ds_allc.rio.reproject('EPSG:4326').rename({'x': 'lon', 'y': 'lat'})
    ds_4326_subset = ds_4326_subset.where(ds_4326_subset['lon'] >= bounds[0], drop = True).where(ds_4326_subset['lon'] <= bounds[2], drop = True).where(ds_4326_subset['lat'] >= bounds[1], drop = True).where(ds_4326_subset['lat'] <= bounds[3], drop = True)

    # reproject with remapping
    #lon = np.arange(bounds[0], bounds[2] + resolution, resolution) # only needed to reproject with remapping
    #lat = np.arange(bounds[1], bounds[3] + resolution, resolution) # only needed to reproject with remapping
    #ds_4326_subset = xr.Dataset(coords = {'time': ds_allc['time'], 'lat': lat, 'lon': lon})
    #ds_4326_subset = ds_4326_subset.rio.write_crs('EPSG:4326').rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat')
    #ds_4326_subset = ds_allc.rio.reproject_match(ds_4326_subset).rename({'x': 'lon', 'y': 'lat'})

    comp = dict(zlib = True, complevel = 5) # netcdf compression at a default level to save storage
    encoding = {var: comp for var in ds_4326_subset.data_vars}
    ds_4326_subset.to_netcdf(ncfile, encoding = encoding)
    ds_4326_subset.close()

    return

if __name__ == '__main__':
    year = 2003
    month = 1
    bounds = [73.375, 22.4375, 97.8125, 31.5]
    resolution = 0.0625 # only needed to reproject with remapping

    if len(sys.argv) > 1: year, month = int(sys.argv[1][:4]), int(sys.argv[1][4:6])
    if len(sys.argv) > 2: bounds = [float(v) for v in sys.argv[2].split(',')]
    if len(sys.argv) > 3: resolution = int(sys.argv[3])
    
    ncfile = 'HMA_DM_6H_v01_subset_{}{:02d}.nc4'.format(year, month) # output netcdf filename
    subset(year, month, bounds, resolution, ncfile)