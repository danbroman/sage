#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 21:43:26 2021

@author: fabianl
"""



import numpy as np
import pandas as pd # times series
import xarray as xr
import shapefile
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from scipy.stats import rankdata
import matplotlib
import pyproj 
import cartopy as cpy
# matplotlib.use('agg')
import os 
try: 
    os.nice(5-os.nice(0)) # set current nice level to 5, if it is lower 
except: # nice level already above 5
    pass



layering_coef =   [10,100]

crs=cpy.crs.epsg(5266)

BhutanGrid=pyproj.Proj("EPSG:5266")
WGS84=pyproj.Proj("EPSG:4326") # LatLon with WGS84 



# %%  ###  Non linear Model fit

from lmfit import Model
from lmfit.models import LinearModel

#### Define model:
lin_mod = LinearModel(prefix='line_') # Linear Model already in LMfit

def jump(x, h00, aa):
    """heaviside step function"""
    o = np.zeros(len(x))
    try:
        imid = min(np.where(x >= h00)[0])
        o[0:imid] = aa
    except ValueError: # if h00 is smaller than the lowest x value
        o = np.zeros(len(x))
        
    return o

def rectangle_cos(x, h0, delta_h, a):
    """rectangle function combined with a cosinus function"""
    o = np.zeros(len(x))
    try:
        imid_right = max(np.where(x <= h0 +delta_h)[0])
        imid_left = min(np.where(x >= h0)[0])
        o[imid_left:imid_right] = a/2 * (1+np.cos(np.pi * (x[imid_left:imid_right]-h0)/(delta_h)))
    except ValueError: # if h00 is smaller than the lowest x value
        o = np.zeros(len(x))
    return o    
  
# Model: 
mod = lin_mod -  Model(rectangle_cos) - Model(jump)


#%%
def border_plot():
    # borders plot
    sf = shapefile.Reader("/nas8/VanishingTreasures/data/GIS/BTN_adm/BTN_adm1_EPSG5266.shp",encoding="latin1")
    for shape in sf.shapeRecords():
        if len(shape.shape.parts) == 1:
            x = [i[0] for i in shape.shape.points]             
            y = [i[1] for i in shape.shape.points]
            plt.plot(x,y,'gray')
        else:
            for index in range(len(shape.shape.parts)):
                try:
                    x = [i[0] for i in shape.shape.points[shape.shape.parts[index]:shape.shape.parts[index+1]]]
                    y = [i[1] for i in shape.shape.points[shape.shape.parts[index]:shape.shape.parts[index+1]]]
                except:
                    x = [i[0] for i in shape.shape.points[shape.shape.parts[index]:]]
                    y = [i[1] for i in shape.shape.points[shape.shape.parts[index]:]]
                plt.plot(x,y,'gray')
                
def plot_letter(letter):
    plt.text(-0.04,1.06,letter, transform=plt.gca().transAxes, fontweight='bold', fontsize='large') 

    

DS = xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/mask_bhutan.nc')
mask= DS.Band1 

#%% ERA5 
DS=xr.open_mfdataset('/hp8/Ecmwf/Era5/Bhutan/TA_*_bhutan.nc', concat_dim='p', combine='nested')
T_era5_min = DS.T.load().resample(t='D').min().mean(axis=(2,3))-273.15
T_era5_max = DS.T.load().resample(t='D').max().mean(axis=(2,3))-273.15

DS=xr.open_mfdataset('/hp8/Ecmwf/Era5/Bhutan/HGT_*_bhutan.nc', concat_dim='p', combine='nested')
HGT_era5_mean = (DS.Z.load().resample(t='D').mean().mean(axis=(2,3))/9.81)


#%% 

# # read metadata


### ==================  generalized distances calculated according to Frei (2014) ========================
DS=xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/distances/distances_EPSG5266.nc')
distances=DS.distances
station_names_distances = DS.station_names.load()

stations =  pd.read_csv('/nas8/VanishingTreasures/data/Corrected_metadata.csv',  decimal='.', index_col=0) 
WGS84=pyproj.Proj("EPSG:4326") # LatLon with WGS84 
BhutanLambert=pyproj.Proj("EPSG:5266")  # projection of Bhutan
transformer = pyproj.Transformer.from_proj(WGS84,BhutanLambert)
stations_x,stations_y = transformer.transform(np.array(stations.latitude),np.array(stations.longitude)) 


# topo 
DS=xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/topos/topos_average_1kmEPSG5266.nc')
topo=DS.Band1
topo_1d=topo.values.flatten()




for var in ['t_max',  't_min']:
    print(var)
    if var=='t_min':
        var_cf_conform = 'tasmin'
        min_plot = -16
        max_plot = 27
    elif var=='t_max':
        var_cf_conform = 'tasmax'
        min_plot = -4
        max_plot = 36
        
    ### ==========================  distances =================================
    DS=xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/distances/distances_EPSG5266.nc')

    # read met data
    df = pd.read_csv('/nas8/VanishingTreasures/data/'+var+'_outliers_removed_trange_MonthlyHomogenized_breaks_removed_trange.csv', index_col=0, parse_dates=True)
    distances=DS.distances
    station_names_distances = DS.station_names.load()

    # monthly means
    df_clim_months = df.groupby(df.index.month).mean()
    

     
    # %% interpolation
    df = df[slice('1996','2020')]
    TEMPERATURE=np.full([len(df), np.size(topo,0),np.size(topo,1)], np.nan, dtype='f4')
    
    ###  non linear height dependency 
    for day in range(len(df)): 

        
        nans=np.isnan(df.iloc[day,:].values)
        
        print(str(df.index[day])[0:10])
        T_today =  df.iloc[day,:].values[~nans]
        x=stations.elevation_m[~nans]
        # sort vectors with ascending height: 
        x, T_today = zip(*sorted(zip(x, T_today))) # result as tuple! 
        x=np.array(x) 
        T_today=np.array(T_today)
    
        
        # first guess of parameters
        pars = mod.make_params(line_intercept=np.nanmax(T_today), line_slope=-0.007, h0=1000, delta_h=1000, a=3)
        
        # other constraints and conditions for parameters: 
        pars['line_intercept'].set(min=-5, max=45)
        pars['line_slope'].set(min=-0.01, max=0) # lapse rate
        pars['h0'].set(min=0, max=1600) # lower elevation of inversion
        pars['h00'].set(expr='h0')
        pars['delta_h'].set(min=500, max=1500)    # inversion height
        pars['a'].set(min=0, max=5) # inversion strength in °C 
        pars['aa'].set(expr='a')        
    
        # fit this model to data array 
        result = mod.fit(T_today, pars, x=x)
        


        #%%  # Make first guess of temperature grid
    
        ''' =====================================================
            Complicated workaround with interpolation, because the prediction with real topography does not work
            I tried predicting each grid line separately and in ascending order, but nothing worked 
            ===================================================== '''
        z=np.arange(0, topo.max()+100)
        prediction= result.eval(result.params, x=z)
        
        # print((np.float32(result.params['line_slope'].value)*100).round(3))
        
        mapping = interp1d(z, prediction, kind='linear', bounds_error=False)
        first_guess_1d = mapping(topo.values.flatten())
                    
        first_guess=np.reshape(first_guess_1d,(np.size(topo,0),np.size(topo,1)))
        
        #%% correct high elevations with ERA5 temperature lapse rate
        lapse_rate= result.params['line_slope'].value
        
        if var=='t_max': 
            t_400 = T_era5_max.sel(p=400, t=str(df.index[day])[0:10]).values
            t_500 = T_era5_max.sel(p=500, t=str(df.index[day])[0:10]).values
            t_600 = T_era5_max.sel(p=600, t=str(df.index[day])[0:10]).values
            t_700 = T_era5_max.sel(p=700, t=str(df.index[day])[0:10]).values
        elif var=='t_min':
            t_400 = T_era5_min.sel(p=400, t=str(df.index[day])[0:10]).values
            t_500 = T_era5_min.sel(p=500, t=str(df.index[day])[0:10]).values
            t_600 = T_era5_min.sel(p=600, t=str(df.index[day])[0:10]).values
            t_700 = T_era5_min.sel(p=700, t=str(df.index[day])[0:10]).values
        HGT400= HGT_era5_mean.sel(p=400, t=str(df.index[day])[0:10]).values 
        HGT500= HGT_era5_mean.sel(p=500, t=str(df.index[day])[0:10]).values
        HGT600= HGT_era5_mean.sel(p=600, t=str(df.index[day])[0:10]).values 
        HGT700= HGT_era5_mean.sel(p=700, t=str(df.index[day])[0:10]).values 
        
        deltaH650 = HGT600 - HGT700 
        deltaH550 = HGT500 - HGT600 
        deltaH450 = HGT400 - HGT500
         
        lapse_rate_era5_650 = (t_600- t_700) / deltaH650
        lapse_rate_era5_550 = (t_500- t_600) / deltaH550
        lapse_rate_era5_450 = (t_400- t_500) / deltaH450
        
        
        ##############  from 600 hPa  #################################
        first_guess[(topo.values>HGT600) & (topo.values<HGT500)] = mapping(HGT600) + (topo.values[(topo.values>HGT600) & (topo.values<HGT500)] -HGT600) * lapse_rate_era5_550
        new500T = mapping(HGT600) + (HGT500 - HGT600) * lapse_rate_era5_550     
        
        ## above 500 hPa 
        first_guess[(topo.values>=HGT500)] = new500T + (topo.values[topo.values>=HGT500] -HGT500) * lapse_rate_era5_450
        
        
        
  
        

        first_guess = xr.DataArray(first_guess, coords=[DS.y, DS.x], name='temperature')        

        #%% Residuals 
        # inaccurate residuals (because grid cells usually have different elevations than the stations)
        # residuals_stations = df.iloc[day,:] - first_guess.sel(lat=xr.DataArray(stations.latitude, dims="points"), lon=xr.DataArray(stations.longitude, dims="points"), method="nearest")
        
        # residuals with exact station elevation
        residuals_stations = df.iloc[day,:] - mapping(stations.elevation_m)
 
    
        avail_stations = np.where(~np.isnan(df.iloc[day]))[0]
        residuals_stations_curr_day = residuals_stations.iloc[avail_stations]
        #residual interpolation
        # layering_coef= [50,50]
        distances_selected= distances.sel(layering_coef=layering_coef, station=avail_stations+1)

        if np.size(distances_selected.layering_coef)>1:
            residual_interp_3d = np.full([len(distances_selected.layering_coef), len(distances_selected.y),  len(distances_selected.x)], np.nan,  dtype='f4')
            for coef in range(len(distances_selected.layering_coef)):
                indices=np.argpartition(distances_selected[coef].values,3, axis=0)[0:4]
                four_distances = np.take_along_axis(distances_selected[coef].values, indices, axis=0)
                four_residuals = residuals_stations_curr_day.values[indices]        
                residual_interp_3d[coef] = 1/np.sum(1/np.square(four_distances), axis=0)*np.sum(four_residuals/np.square(four_distances), axis=0) # Frei (2014), eq. 4
        
            residual_interp = np.mean(residual_interp_3d, axis=0)
        else: 
            indices=np.argpartition(distances_selected[0].values,3, axis=0)[0:4]
            four_distances = np.take_along_axis(distances_selected[0].values, indices, axis=0)
            four_residuals = residuals_stations_curr_day.values[indices]        
            residual_interp = 1/np.sum(1/np.square(four_distances), axis=0)*np.sum(four_residuals/np.square(four_distances), axis=0) # Frei (2014), eq. 4

        TEMPERATURE[day]= first_guess + residual_interp
        
        
    
    
    #%% paper plot
            fig, ax = plt.subplots(2,2, figsize=(1.45*7, 0.78*7))
            # plt.subplot(2,2,1)
            plt.subplot2grid((26,18), (1, 1), rowspan=10, colspan=7)
            plt.plot(T_today, x, 'ko')
            y=result.eval(x=np.linspace(100,int(HGT600),int(HGT600)))
            plt.plot(y, np.linspace(100,int(HGT600),int(HGT600)),  'dimgray')  
            y=mapping(HGT600) + (np.linspace(int(HGT600),5300,int(5300-HGT600)) - HGT600) * lapse_rate_era5_550    
            plt.plot(y, np.linspace(int(HGT600),5300,int(5300-HGT600)),  'dimgray', linestyle='dotted')
            plt.grid(True)
            plt.title(var +' on '+str(df.index[day])[0:10])
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Elevation (m)')
            plt.legend(['Observations','Regression', 'ERA5 lapse rate'])
            plt.text(-0.17,1.07,'a', transform=plt.gca().transAxes, fontweight='bold', fontsize='large') 

            
            
            ax = plt.subplot(2,2,2,  projection=crs)
            ds_plot = xr.DataArray(first_guess, coords=[DS.y, DS.x], name='')
            (ds_plot*mask.values).plot(ax=ax,cmap='nipy_spectral', levels=np.arange(min_plot,max_plot,4)  , extend='both')
            border_plot()
            # pilot_area_plot()
            plt.ylabel(None)
            plt.xlabel(None)
            plt.axis('off')
            plt.text(0.01,0.97, 'Min: '+str(int((ds_plot*mask.values).min()))+' °C. Max: '+str(int((ds_plot*mask.values).max()))+' °C', transform=plt.gca().transAxes) 
            plt.title('Regression (°C)') 
            plot_letter('b')
            
            ax = plt.subplot(2,2,3,  projection=crs)
            ds_plot = xr.DataArray(residual_interp, coords=[DS.y, DS.x])
            (ds_plot*mask.values).plot(ax=ax,cmap='RdBu_r', vmin=-6, vmax=6)
            plt.scatter(stations_x[avail_stations],stations_y[avail_stations],c=residuals_stations_curr_day.values, cmap='RdBu_r' , vmin=-6, vmax=6, edgecolors='k')
            border_plot()
            # pilot_area_plot()
            plt.ylabel(None)
            plt.xlabel(None)
            plt.axis('off')
            plt.text(0.01,0.97, 'Min: '+str(int((ds_plot*mask.values).min()))+' °C. Max: '+str(int((ds_plot*mask.values).max()))+' °C', transform=plt.gca().transAxes) 
            plt.title('Residuals (°C)') 
            plot_letter('c')
            
            ax = plt.subplot(2,2,4,  projection=crs)
            ds_plot = xr.DataArray(TEMPERATURE[day], coords=[DS.y, DS.x])
            (ds_plot*mask.values).plot(ax=ax,cmap='nipy_spectral', levels=np.arange(min_plot,max_plot,4) , extend='both')
            border_plot()
            # pilot_area_plot()
            plt.ylabel(None)
            plt.xlabel(None)
            plt.axis('off')
            plt.text(0.01,0.97, 'Min: '+str(int((ds_plot*mask.values).min()))+' °C. Max: '+str(int((ds_plot*mask.values).max()))+' °C', transform=plt.gca().transAxes) 
            plt.title('Regression + residuals (°C)')             
            plot_letter('d')
    

            
            # plt.tight_layout()
            plt.subplots_adjust(left=0.02, right=0.99,bottom=0.03, top=0.96)
            plt.savefig('/nas8/VanishingTreasures/Paper2/Plots/temperature_interpolation/maps_daily_withERA5_trange/'+var+'_EPSG5266'+str(layering_coef)+'_'+str(df.index[day])[0:10]+'.png',dpi=300) 
            plt.close('all')    
    
    TEMPERATURE = xr.DataArray(TEMPERATURE, dims = ['time', 'y', 'x'], coords=dict(time=(["time"], df.index.normalize()), y=DS.y, x=DS.x), name=var_cf_conform)
    number_stations = xr.DataArray((~np.isnan(df)).sum(axis=1), dims=['time'], coords=dict(time=(["time"], df.index.normalize())), name='available_stations')
    
    DS = xr.merge([TEMPERATURE, number_stations])
    
    #  add lat and lon 
    X,Y = np.meshgrid(DS.x,DS.y)
    transformer = pyproj.Transformer.from_proj(BhutanGrid,WGS84)
    lat,lon = transformer.transform(X,Y)
    DS = DS.assign(lat=xr.DataArray(data=lat, dims=["y", "x"], coords=dict(y=DS.y, x=DS.x) ))
    DS = DS.assign(lon=xr.DataArray(data=lon, dims=["y", "x"], coords=dict(y=DS.y, x=DS.x) ))

####%%%% encoding  
    encoding  = {}
    for cname in DS.coords:
        encoding[cname] = {"_FillValue": None}
    encoding['lat'] = {"_FillValue": None}
    encoding['lon'] = {"_FillValue": None}
    encoding[var_cf_conform] = {"_FillValue": 1e20}
    encoding['time']={"calendar":'standard'}
  
    DS.time['long_name'] = 'time'
    DS.time['standard_name'] = 'time'  # Optional
    DS.x.attrs['long_name'] = 'x-coordinate in projected coordinate system'
    DS.x.attrs['standard_name'] = 'projection_x_coordinate'
    DS.y.attrs['long_name'] = 'y-coordinate in projected coordinate system'
    DS.y.attrs['standard_name'] = 'projection_y_coordinate'
    DS.lon.attrs['units'] = 'degrees_east'
    DS.lon.attrs['standard_name'] = 'longitude'  # Optional
    DS.lon.attrs['long_name'] = 'longitude'    
    DS.lat.attrs['units'] = 'degrees_north'
    DS.lat.attrs['standard_name'] = 'latitude'  # Optional
    DS.lat.attrs['long_name'] = 'latitude'
    DS_topo=xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/topos/topos_average_1kmEPSG5266.nc')
    DS['transverse_mercator'] = (DS_topo.transverse_mercator)    
    
    DS[var_cf_conform].attrs['standard_name'] = 'air_temperature'
    DS[var_cf_conform].attrs['units'] = 'Celsius'
    
    DS.attrs['title'] = 'Vanishing Treasures for Bhutan'
    DS.attrs['institution'] = 'BOKU-Met, Institute of Meteorology, University of Natural Resources and Life Sciences, Vienna, http://www.wau.boku.ac.at/met.html'
    DS.attrs['contact'] = 'Fabian Lehner (fabian.lehner@boku.ac.at), Herbert Formayer (herbert.formayer@boku.ac.at)'
    from datetime import date
    DS.attrs['creation_date'] = str(date.today())
#######################  
    DS.to_netcdf('/nas8/VanishingTreasures/Paper2/data/'+var_cf_conform+'_EPSG5266_'+str(TEMPERATURE.time.dt.year[0].values)+'-'+str(TEMPERATURE.time.dt.year[-1].values)+'_daily_withERA5_trange.nc', mode='w', unlimited_dims='time', encoding=encoding)

