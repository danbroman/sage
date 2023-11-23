import sys, os, glob
import numpy as np
import xarray as xr

''' 
    Princeton - Global Dataset of Meteorological Forcings for Land Surface Modeling:
    https://hydrology.soton.ac.uk/data/pgf/
    https://doi.org/10.1175/JCLI3790.1

    Requried package installations: conda install -c conda-forge numpy xarray dask netCDF4 (scipy)


'''
def subset(year, bounds, resolution, vars, key, ncfile):
    lon = np.arange(bounds[0], bounds[2] + resolution, resolution) # only needed to remap for new grids
    lat = np.arange(bounds[1], bounds[3] + resolution, resolution) # only needed to remap for new grids
    
    dict_da = {}
    for var in vars:
        ds = xr.open_dataset(os.path.join(var, key, '{}_{}_{}-{}.nc'.format(var, key, year, year)), decode_coords = 'all')
        ds = ds.where(ds['lon'] >= bounds[0], drop = True).where(ds['lon'] <= bounds[2], drop = True).where(ds['lat'] >= bounds[1], drop = True).where(ds['lat'] <= bounds[3], drop = True) # no remapping
        #ds = ds.interp(lon = lon, lat = lat, method = 'linear') # remapping
        dict_da[list(ds.data_vars)[0]] = ds[list(ds.data_vars)[0]].copy()
    
    ds_merge = xr.Dataset(dict_da) # merge over variables

    comp = dict(zlib = True, complevel = 5) # netcdf compression at a default level to save storage
    encoding = {var: comp for var in ds_merge.data_vars}
    ds_merge.to_netcdf(ncfile, encoding = encoding)
    ds_merge.close()

    return

if __name__ == '__main__':
    year = 2003
    bounds = [73.375, 22.4375, 97.8125, 31.5] # lon: -180 to 180
    resolution = 0.0625 # only needed to remap for new grids

    vars_3hourly = ['dlwrf', 'dswrf', 'pres', 'shum', 'tas', 'wind']
    vars_daily = ['dlwrf', 'dswrf', 'prcp', 'pres', 'shum', 'tas', 'tmax', 'tmin', 'wind']

    if len(sys.argv) > 1: year = int(sys.argv[1])
    if len(sys.argv) > 2: bounds = [float(v) for v in sys.argv[2].split(',')]
    if len(sys.argv) > 3: resolution = int(sys.argv[3])

    ncfile = 'GMFD_3hourly_subset_{}.nc'.format(year) # output netcdf file name
    subset(year, bounds, resolution, vars_3hourly, '3hourly', ncfile) # process '3hourly' datasets

    ncfile = 'GMFD_daily_subset_{}.nc'.format(year) # output netcdf file name
    subset(year, bounds, resolution, vars_daily, 'daily', ncfile) # process 'daily' datasets