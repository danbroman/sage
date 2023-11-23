import sys, os, glob
import numpy as np
import xarray as xr

''' 
    GloH2O - Multi-Source Weather (MSWX):
    https://www.gloh2o.org/mswx/
    https://doi.org/10.1175/BAMS-D-21-0145.1

    Requried package installations: conda install -c conda-forge numpy xarray dask netCDF4 (scipy)


'''
def subset(year, bounds, resolution, vars, key, ncfile):
    lon = np.arange(bounds[0], bounds[2] + resolution, resolution) # only needed to remap for new grids
    lat = np.arange(bounds[1], bounds[3] + resolution, resolution) # only needed to remap for new grids
    
    for var in vars:
        path_files = os.path.join(var, key, '{}*.nc'.format(year))
        files = glob.glob(path_files); files.sort()
        for f in files:
            ds = xr.open_dataset(f, decode_coords = 'all')
            ds = ds.where(ds['lon'] >= bounds[0], drop = True).where(ds['lon'] <= bounds[2], drop = True).where(ds['lat'] >= bounds[1], drop = True).where(ds['lat'] <= bounds[3], drop = True) # no remapping
            #ds = ds.interp(lon = lon, lat = lat, method = 'linear') # remapping
            ds.to_netcdf(os.path.join(os.path.dirname(f), '_' + os.path.basename(f))) # create temporary subsetted netcdf files with a prefix of '_'
            ds.close()

        path_files = os.path.join(var, key, '_{}*.nc'.format(year))
        files = glob.glob(path_files); files.sort()
        ds = xr.open_mfdataset(files, chunks = {'time': 8}, concat_dim = 'time', combine = 'nested', decode_coords = 'all', parallel = True)
        ds.to_netcdf('_' + ncfile.replace('.nc', '_{}.nc'.format(var))) # for each variable, create temporary subsetted & concatenated netcdf files with a prefix of '_'
        ds.close()
        for f in files:
            if os.path.isfile(f): os.remove(f) # delete the temporary subsetted netcdf files
    
    list_da = [xr.open_dataset('_' + ncfile.replace('.nc', '_{}.nc'.format(var)), decode_coords = 'all') for var in vars]
    if 'Tmax' in vars: list_da[vars.index('Tmax')] = list_da[vars.index('Tmax')].rename({'air_temperature': 'air_temperature_max'}) # Tmax and Tmin have the same variable name with Temp: air_temperature
    if 'Tmin' in vars: list_da[vars.index('Tmin')] = list_da[vars.index('Tmin')].rename({'air_temperature': 'air_temperature_min'})
    ds_merge = xr.merge(list_da, combine_attrs = 'drop_conflicts') # merge over variables

    comp = dict(zlib = True, complevel = 5) # netcdf compression at a default level to save storage
    encoding = {var: comp for var in ds_merge.data_vars}
    ds_merge.to_netcdf(ncfile, encoding = encoding)
    ds_merge.close()

    for f in glob.glob('_' + ncfile.replace('.nc', '_*.nc')): os.remove(f) # delete the temporary concatenated netcdf files
    
    return

if __name__ == '__main__':
    year = 2003
    bounds = [73.375, 22.4375, 97.8125, 31.5]
    resolution = 0.0625 # only needed to remap for new grids

    vars_3hourly = ['LWd', 'P', 'Pres', 'RelHum', 'SpecHum', 'SWd', 'Temp', 'Wind']
    vars_daily = ['LWd', 'P', 'Pres', 'RelHum', 'SpecHum', 'SWd', 'Temp', 'Wind', 'Tmax', 'Tmin']

    if len(sys.argv) > 1: year = int(sys.argv[1])
    if len(sys.argv) > 2: bounds = [float(v) for v in sys.argv[2].split(',')]
    if len(sys.argv) > 3: resolution = int(sys.argv[3])

    ncfile = 'MSWX-Past_3hourly_subset_{}.nc'.format(year) # output netcdf file name
    subset(year, bounds, resolution, vars_3hourly, '3hourly', ncfile) # process '3hourly' datasets

    ncfile = 'MSWX-Past_daily_subset_{}.nc'.format(year) # output netcdf file name
    subset(year, bounds, resolution, vars_daily, 'Daily', ncfile) # process 'Daily' datasets