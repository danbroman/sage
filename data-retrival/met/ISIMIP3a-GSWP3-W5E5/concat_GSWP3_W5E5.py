import sys, os, glob
import numpy as np
import xarray as xr

''' 
    ISIMIP3a Atmospheric Climate Input Data:
    https://data.isimip.org/10.48364/ISIMIP.982724.2
    https://doi.org/10.48364/ISIMIP.982724.2

    Requried package installations: conda install -c conda-forge numpy xarray dask netCDF4 (scipy)


'''
def concat(year, bounds, resolution, vars, ncfile):
    lon = np.arange(bounds[0], bounds[2] + resolution, resolution) # only needed to remap for new grids
    lat = np.arange(bounds[1], bounds[3] + resolution, resolution) # only needed to remap for new grids
    
    dict_da = {}
    for var in vars:
        ds = xr.open_dataset(glob.glob(os.path.join(var, 'gswp3-w5e5_counterclim_{}_*_{}_{}.nc'.format(var, year, min(year + 9, 2019))))[0], decode_coords = 'all')
        #ds = ds.interp(lon = lon, lat = lat, method = 'linear') # remapping
        dict_da[list(ds.data_vars)[0]] = ds[list(ds.data_vars)[0]].copy()
    
    ds_merge = xr.Dataset(dict_da) # merge over variables

    comp = dict(zlib = True, complevel = 5) # netcdf compression at a default level to save storage
    encoding = {var: comp for var in ds_merge.data_vars}
    ds_merge.to_netcdf(ncfile, encoding = encoding)
    ds_merge.close()

    return

if __name__ == '__main__':
    year = 2001 # 10-year interval, e.g., 2001-2010
    bounds = [73.375, 22.4375, 97.8125, 31.5] # only needed to remap for new grids
    resolution = 0.0625 # only needed to remap for new grids

    vars_daily = ['hurs', 'huss', 'pr', 'ps', 'rlds', 'rsds', 'sfcwind', 'tas', 'tasmax', 'tasmin']

    if len(sys.argv) > 1: year = int(sys.argv[1])
    if len(sys.argv) > 2: bounds = [float(v) for v in sys.argv[2].split(',')]
    if len(sys.argv) > 3: resolution = int(sys.argv[3])

    ncfile = 'GSWP3-W5E5_subset_{}_{}.nc'.format(year, year + 9) # output netcdf file name
    concat(year, bounds, resolution, vars_daily, ncfile)