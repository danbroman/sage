import os, glob, functools
import numpy as np
import pandas as pd
import xarray as xr

xr.set_options(keep_attrs = True)

list_models = ['ERA5', 'ERA5-L', 'MSWX-P', 'GSWP3-W5E5', 'HMA', 'GMFD'] # a list of model variables
models_path = {
    'ERA5': 'ECMWF-ERA5',
    'ERA5-L': 'ECMWF-ERA5-Land',
    'MSWX-P': 'GloH2O-MSWX-Past',
    'GSWP3-W5E5': 'ISIMIP3a-GSWP3-W5E5',
    'HMA': 'NASA-HMA',
    'GMFD': 'Princeton-GMFD',
} # filepath to load for a selected model
models_fprefix = {
    'ERA5': 'ERA5_sl_subset_{}*.nc',
    'ERA5-L': 'ERA5-Land_subset_{}*.nc',
    'MSWX-P': 'MSWX-Past_daily_subset_{}*.nc',
    'GSWP3-W5E5': 'GSWP3-W5E5_subset_{}*.nc',
    'HMA': 'HMA_DM_6H_v01_subset_{}*.nc4',
    'GMFD': 'GMFD_daily_subset_{}*.nc',
} # filename to load for a selected model
models_name = {
    'ERA5': 'ECMWF ERA5',
    'ERA5-L': 'ECMWF ERA5-LAND',
    'MSWX-P': 'GloH2O MSWX-Past',
    'GSWP3-W5E5': 'ISIMIP3a GSWP3-W5E5',
    'HMA': 'NASA HMA',
    'GMFD': 'Princeton GMFD',
} # name to display for a selected model

# function to subset a dataset by x/y or lon/lat
def _preprocess_subset(ds, vars = None, bounds = None):
    if vars is not None: ds = ds[vars]
    if bounds is not None:
        if 'lon' in ds.dims and 'lat' in ds.dims: x, y = 'lon', 'lat'
        elif 'longitude' in ds.dims and 'latitude' in ds.dims: x, y = 'longitude', 'latitude'
        elif 'x' in ds.dims and 'y' in ds.dims: x, y = 'x', 'y'
        else: return ds
        ds = ds.where(ds[x] >= bounds[0], drop = True).where(ds[x] <= bounds[2], drop = True).where(ds[y] >= bounds[1], drop = True).where(ds[y] <= bounds[3], drop = True)
    return ds

# function to read a dataset
def read_datasets(path, model, years = None, bounds = None, vars_list = None, save_nc = None):
    if years is None: # read all the files
        path_iter = os.path.join(path, models_path[model], models_fprefix[model].format(''))
        files = glob.glob(path_iter)    
    else:
        files = []
        if model == 'GSWP3-W5E5': files += glob.glob(os.path.join(path, models_path[model], models_fprefix[model].format(''))) # GSWP3-W5E5 has files with a 10-year interval
        else:
            for y in years:
                path_iter = glob.glob(os.path.join(path, models_path[model], models_fprefix[model].format(y)))
                files += path_iter
    files.sort()

    func_preprocess = functools.partial(_preprocess_subset, vars = vars_list, bounds = bounds) # apply the subset function before concatenating
    if len(files) > 0: ds = xr.open_mfdataset(files, concat_dim = 'time', combine = 'nested', decode_coords = 'all', preprocess = func_preprocess, parallel = True)
    else: return None

    if save_nc is not None:
        if isinstance(save_nc, str): ds.to_netcdf(save_nc)

    return ds

# function to convert a pricipitation variable into total precipitation per step
def convert_precipitation(da_prcp, model):
    da = da_prcp.copy()
    if model == 'ERA5':
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
        # Accumulations are over the hour (the processing period) ending at the validity date/time
        da['time'] = da['time'] - pd.Timedelta(hours = 1)
        da = da.sel(time = da['time'][1:])
        da = da.reindex(time = da_prcp['time']) * 1000 # m to mm/step(hourly)
    elif model == 'ERA5-L':
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
        # Accumulations are from 00 UTC to the hour ending at the forecast step
        da = da.diff(dim = 'time')
        da['time'] = da['time'] - pd.Timedelta(hours = 1)
        da.loc[::24,:,:] = da_prcp.roll(time = -1).resample(time = '1D').nearest()
        da = da.reindex(time = da_prcp['time']) * 1000 # m to mm/step(hourly)
    elif model in ['GSWP3-W5E5', 'GMFD']: da = da * 24 * 60 * 60 # kg m-2 s-1 to mm/step(daily)
    elif model == 'HMA': da = da * 6 # mm/h to mm/step(6-hourly)
    da.attrs['units'] = 'mm'

    return da

# function to convert a temperature variable into degreeC
def convert_temperature(da_temp, model):
    da = da_temp.copy()
    if model in ['ERA5', 'ERA5-L', 'GSWP3-W5E5', 'HMA', 'GMFD']: da = da - 273.15 # Kelvin to degreeC
    da.attrs['units'] = 'degC'

    return da

if __name__ == '__main__':
    path_data = 'datasets'
    #years = np.arange(1996, 2017)
    years = np.arange(2003, 2005)
    vars_list = {
        'ERA5': [
#            'd2m', # 2 metre dewpoint temperature
#            'sf', # Snowfall
#            'sp', # Surface pressure
#            'ssr', # surface net short-wave (solar) radiation
#            'ssrd', # Surface short-wave (solar) radiation downwards
#            'strd', # Surface long-wave (thermal) radiation downwards
            't2m', # 2 metre temperature
            'tp', # Total precipitation
#            'u10', # 10 metre U wind component
#            'v10', # 10 metre V wind component
        ],
        'ERA5-L': [
#            'd2m', # 2 metre dewpoint temperature
#            'sf', # Snowfall
#            'sp', # Surface pressure
#            'ssr', # surface net short-wave (solar) radiation
#            'ssrd', # Surface short-wave (solar) radiation downwards
#            'strd', # Surface long-wave (thermal) radiation downwards
            't2m', # 2 metre temperature
            'tp', # Total precipitation
#            'u10', # 10 metre U wind component
#            'v10', # 10 metre V wind component
        ],
        'MSWX-P': [
            'air_temperature',
            'air_temperature_max',
            'air_temperature_min',
#            'downward_longwave_radiation',
#            'downward_shortwave_radiation',
            'precipitation',
#            'relative_humidity',
#            'specific_humidity',
#            'surface_pressure',
#            'wind_speed',
        ],
        'GSWP3-W5E5': [
#            'hurs', # Near-Surface Relative Humidity
#            'huss', # Near-Surface Specific Humidity
            'pr', # Precipitation
#            'ps', # Surface Air Pressure
#            'rlds', # Surface Downwelling Longwave Radiation
#            'rsds', # Surface Downwelling Shortwave Radiation
#            'sfcwind', # Near-Surface Wind Speed
            'tas', # Near-Surface Air Temperature
            'tasmax', # Near-Surface Air Temperature
            'tasmin', # Near-Surface Air Temperature
        ],
        'HMA': [
#            'LRad', # Downscaled ECMWF surface incident longwave radiation
            'P_CHP', # Downscaled CHIRPS total precipitation
            'P_ECMWF', # Downscaled ECMWF total precipitation
#            'Pair', # Downscaled ECMWF surface pressure
#            'RHum', # Downscaled ECMWF near-surface relative humidity
#            'SHum', # Downscaled ECMWF near-surface specific humidity
#            'SRad', # Downscaled ECMWF surface incident shortwave radiation
            'Tair', # Downscaled ECMWF near-surface temperature
#            'Tdew', # Downscaled ECMWF near-surface dew point temperature
#            'Wspd', # Downscaled ECMWF near-surface total wind speed
        ],
        'GMFD': [
#            'dlwrf', # Downward longwave radiation
#            'dswrf', # Downward shortwave radiation
            'prcp', # Precipitation
#            'pres', # daily surface pressure
#            'shum', # daily specific humidity
            'tas', # Air temperature
            'tmax', # Maximum air temperature
            'tmin', # Minimum air temperature
#            'wind', # Windspeed
        ],
    }
    bounds = [88.6875, 25.5, 93.375, 28.625]
    
    save_nc = {
        'ERA5': '_' + models_fprefix['ERA5'].format('test').replace('*', ''),
        'ERA5-L': '_' + models_fprefix['ERA5-L'].format('test').replace('*', ''),
        'MSWX-P': '_' + models_fprefix['MSWX-P'].format('test').replace('*', ''),
        'GSWP3-W5E5': '_' + models_fprefix['GSWP3-W5E5'].format('test').replace('*', ''),
        'HMA': None, # '_' + models_fprefix['HMA'].format('test').replace('*', ''),
        'GMFD': '_' + models_fprefix['GMFD'].format('test').replace('*', ''),
    }
    key_prcp = {'ERA5': ['tp'], 'ERA5-L': ['tp'], 'MSWX-P': ['precipitation'], 'GSWP3-W5E5': ['pr'], 'HMA': ['P_CHP', 'P_ECMWF'], 'GMFD': ['prcp']}
    list_ds = []
    for m in list_models:
        ds = read_datasets(path = path_data, model = m, years = years, bounds = bounds, vars_list = vars_list[m], save_nc = save_nc[m])
        for key in key_prcp[m]: ds[key] = convert_precipitation(da_prcp = ds[key], model = m)
        list_ds.append(ds)