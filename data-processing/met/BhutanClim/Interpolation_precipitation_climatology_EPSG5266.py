#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:21:34 2021

@author: fabianl
"""
import numpy as np
import pandas as pd # times series
import xarray as xr
import shapefile
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import colors
import nclcmaps
import pyproj 
import cartopy as cpy
from gstools import Gaussian, krige
import gstools as gs
# matplotlib.use('agg')
import os 
try: 
    os.nice(5-os.nice(0)) # set current nice level to 5, if it is lower 
except: # nice level already above 5
    pass


''''  PRECIPITATION INTERPOLATION 2 STEPS:
    
    LIN REGRESSION
    RESIDUAL INTERPOLATION
    
    '''

# border between north and south in °
border_degree=27.15

# exponent to apply on data before linear regression
exp=1

# additive or multiplicative residuals
method='multiplicative' # 'multiplicative'    'additive' 

method_interpolation = 'IDWsquared'   #'OK', 'IDWsquared'

# layering coefficient to multiply elevation (Frei et al., 2014)
# if a 1D array is provided, a mean of the different residuals is calculated
# layering_coef = [5,10,15,20] # 1,5,10,15,20,30,40,50 , [10,15,20]

layering_coef = [1,10,30,50]

colormap = nclcmaps.cmap('precip_11lev') 

extent_transition_zone = 40000 # m 

# # do not show plots? 
# plt.ioff()

#%%
# crs=cpy.crs.epsg(5266)


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
    plt.text(-0.06,1.06,letter, transform=plt.gca().transAxes, fontweight='bold', fontsize='large') 
            
DS = xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/mask_bhutan.nc')
mask= DS.Band1   
            

#%% 


### ==========================  distances =================================
DS=xr.open_dataset('/nas8/VanishingTreasures/Paper2/data/distances/distances_EPSG5266_euclidian.nc')
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

# read met data
df = pd.read_csv('/nas8/VanishingTreasures/data/rainfall_outliers_removed_homogenized.csv', index_col=0, parse_dates=True)



# %% monthly interpolation on map 
p_predict = topo.values.reshape(-1, 1)
xx, yy = np.meshgrid(DS.x.values, DS.y.values)
x_predict = np.array([yy.flatten(), xx.flatten()]).T

# weight for merging the subregions: 
border=transformer.transform(np.array(border_degree),np.array(90.5))[1]

weight = (DS.y-border+extent_transition_zone/2) /extent_transition_zone
weight[weight<0]=0
weight[weight>1]=1
# weight.plot()
weight_north=np.tile(weight.values, [len(DS.x),1]).T
weight_south=abs(1-weight_north)   

month_len = [31,28,31,30,31,30,31,31,30,31,30,31]


# monthly means
df_clim_months = df.groupby(df.index.month).mean()*np.array(month_len)[:,None]


linreg_all=np.empty([12, len(DS.y), len(DS.x) ], dtype='f4')
result_all=np.empty([12, len(DS.y), len(DS.x) ], dtype='f4')

  
for month in range(0,12):
    print('month= '+str(month+1))

    south_bool = stations.latitude<border_degree
    north_bool = stations.latitude>=border_degree
    bool_list = [south_bool, north_bool]
    result=np.empty([len(bool_list), len(p_predict)], dtype='f4')
    coefs = np.empty(2)
    intercepts=np.empty(2)

    for subregion in range(2):
        bool_vector = bool_list[subregion]
        
        # Linear Regression
        p = stations.elevation_m.values[bool_vector].reshape(-1, 1)
        # x = np.array([stations_y[bool_vector], stations_x[bool_vector]]).T
        df_clim_current = df_clim_months.iloc[month,:][bool_vector]
        

        
        reg = LinearRegression().fit(p, df_clim_current**exp)  
        print(reg.score(p, df_clim_current**exp).astype('f2'))
        # print(np.corrcoef(stations.elevation_m[bool_vector], df_clim_current)[0,1])
        coefs[subregion]= reg.coef_[0]
        intercepts[subregion]= reg.intercept_
           
        # trick to avoid extreme extrapolation
        p_predict_cut = np.copy(p_predict)
        p_predict_cut[p_predict_cut>np.max(p)] = p_predict_cut[p_predict_cut>np.max(p)] - (p_predict_cut[p_predict_cut>np.max(p)]-np.max(p))*0.75
        
        # predict
        result[subregion,:] = reg.predict(p_predict_cut)**(1/exp)
        # correct for wrong interpolation 
        result[result<1]=1
    
    result_reshaped=result.reshape(2,len(DS.y), len(DS.x))
    
    linreg_all[month,:,:] = (result_reshaped[0,:,:]*weight_south + result_reshaped[1,:,:]*weight_north)
      
    plt.subplots(5,3, figsize=(1.00*7, 2.280*7))
    plt_text = ['South', 'North', 'Combined']
    hlines = np.array([border-extent_transition_zone/2, border, border+extent_transition_zone/2])
    # %  plot    
    for plot_nr in range(3):     
    # plot with stations

        if plot_nr<2:
            plt.subplot(5,3,(3*plot_nr+1,3*plot_nr+2))
            DS['rainfall'] = xr.DataArray(result_reshaped[plot_nr], coords=[DS.y, DS.x])
            (DS.rainfall*mask.values).plot(cmap='nipy_spectral_r', levels=np.arange(1,10,0.2)**3 , add_colorbar=False)
            plt.hlines(hlines,DS.x.min(), DS.x.max(), linestyles='--', colors=['silver', 'k', 'silver'], alpha=0.6)
        elif plot_nr==2:
            plt.subplot(5,1,plot_nr+1)
            DS['rainfall'] = xr.DataArray(linreg_all[month,:,:], coords=[DS.y, DS.x])
            (DS.rainfall*mask.values).plot(cmap='nipy_spectral_r', levels=np.arange(1,10,0.2)**3)
        plt.axis('off')
            

        plt.ylabel(None)
        plt.xlabel(None)
        border_plot()
        plt.text(0.01,0.97, 'Min: '+str(int(DS['rainfall'].min()))+' mm. Max: '+str(int(DS['rainfall'].max()))+' mm', transform=plt.gca().transAxes) 
        # plt.annotate('Min: '+str(int(DS['rainfall'].min()))+' mm. Max: '+str(int(DS['rainfall'].max()))+' mm', xy=(88.8, 28.3) )

        if plot_nr<2:  
            if exp!=1:
                plt.title('Coef: '+str(round(coefs[plot_nr],4))+'mm/m^'+str(exp)+', intercept: '+str(round(intercepts[plot_nr],1))+'mm^'+str(exp))  
            else:
                plt.title('Coef: '+str(round(coefs[plot_nr],4))+'mm/m, intercept: '+str(round(intercepts[plot_nr],1))+'mm.')
        elif plot_nr>=2:            
            plt.title('Lin. reg., north and south combined (mm)') 
         

    plt.subplot(5,3,3)
    x_min = min(stations.elevation_m.values[bool_list[0]])
    x_max = max(stations.elevation_m.values[bool_list[0]])
    plt.scatter( stations.elevation_m.values[bool_list[0]], df_clim_months.iloc[month,:][bool_list[0]])
    plt.plot(np.linspace(x_min,x_max), (intercepts[0] + np.linspace(x_min,x_max)*(coefs[0]))**(1/exp), 'k')
    plt.xlabel('elevation (m)')
    plt.title('South of '+str(border_degree)+'°')
    plt.grid()
    
    plt.subplot(5,3,6)
    x_min = min(stations.elevation_m.values[bool_list[1]])
    x_max = max(stations.elevation_m.values[bool_list[1]])
    plt.scatter( stations.elevation_m.values[bool_list[1]], df_clim_months.iloc[month,:][bool_list[1]])
    plt.plot(np.linspace(x_min,x_max), (intercepts[1] + np.linspace(x_min,x_max)*(coefs[1]))**(1/exp), 'k')
    plt.xlabel('elevation (m)')
    plt.title('North of '+str(border_degree)+'°')
    plt.grid()

        
       
    plt.suptitle('Precipitation month '+str(month+1)+' (mm)', y=0.999)         

    # Residuals 
    if method=='additive':
        residuals_stations = df_clim_months.iloc[month,:].values - DS['rainfall'].sel(y=xr.DataArray(stations_y, dims="points"), x=xr.DataArray(stations_x, dims="points"), method="nearest")
    elif method=='multiplicative':
        residuals_stations = df_clim_months.iloc[month,:].values / DS['rainfall'].sel(y=xr.DataArray(stations_y, dims="points"), x=xr.DataArray(stations_x, dims="points"), method="nearest")
        

        
    
    #%%###===================  inverse distance like SPARTACUS    ===================
    if method_interpolation=='IDWsquared':
        # ## get indices for the four nearest stations per grid cell
        distances_selected= distances.sel(layering_coef=layering_coef)
        if np.size(distances_selected.layering_coef)>1:
            residual_interp_3d = np.full([len(distances_selected.layering_coef), len(distances_selected.y),  len(distances_selected.x)], np.nan,  dtype='f4')
            for coef in range(len(distances_selected.layering_coef)):
                indices=np.argpartition(distances_selected[coef].values,3, axis=0)[0:4]
                four_distances = np.take_along_axis(distances_selected[coef].values, indices, axis=0)
                four_residuals = residuals_stations.values[indices]        
                residual_interp_3d[coef] = 1/np.sum(1/np.square(four_distances), axis=0)*np.sum(four_residuals/np.square(four_distances), axis=0) # Frei (2014), eq. 4
        
            residual_interp = np.mean(residual_interp_3d, axis=0)
        else: 
            indices=np.argpartition(distances_selected.values,3, axis=0)[0:4]
            four_distances = np.take_along_axis(distances_selected.values, indices, axis=0)
            four_residuals = residuals_stations.values[indices]        
            residual_interp = 1/np.sum(1/np.square(four_distances), axis=0)*np.sum(four_residuals/np.square(four_distances), axis=0) # Frei (2014), eq. 4
        
    if method_interpolation=='OK':
        sigma = residuals_stations.mean().values
        data_norm = (residuals_stations.values-1)/sigma
        bin_center, gamma = gs.vario_estimate((stations_x, stations_y), data_norm)
        
        fit_model = gs.JBessel(dim=2)
        para, pcov = fit_model.fit_variogram(bin_center, gamma)
        krig = krige.Ordinary(fit_model, cond_pos=(stations_x, stations_y),cond_val=data_norm)
        field,krige_var =  krig((DS.x.values, DS.y.values), mesh_type='structured')
        
        residual_interp = (field.T*sigma+1)
    
    ####===================  =================    ===================
    if method=='additive':
        result_all[month,:,:]= linreg_all[month,:,:] + residual_interp
    elif method=='multiplicative':
        result_all[month,:,:]= linreg_all[month,:,:] * residual_interp
    
    # limit values to 90% of driest station for each month
    result_all[month,:,:][result_all[month,:,:]<df_clim_months.iloc[month,:].min()*0.9] = df_clim_months.iloc[month,:].min()*0.9
    
    if method=='multiplicative':
        vmax = 2
        vmin = 0
    elif max(abs(residuals_stations.values))<100 and method=='additive':
        vmax = 40
        vmin=-vmax
    else:
        vmax=150
        vmin=-vmax
    
    plt.subplot(5,1,4)
    DS['rainfall'] = xr.DataArray(residual_interp, coords=[DS.y, DS.x])
    (DS.rainfall*mask.values).plot(cmap='BrBG' , vmin=vmin, vmax=vmax)
    border_plot()
    plt.ylabel(None)
    plt.xlabel(None)
    plt.text(0.01,0.97, 'Min: '+str(int((DS['rainfall']*mask.values).min()))+'. Max: '+str(int((DS['rainfall']*mask.values).max()))+'', transform=plt.gca().transAxes) 
    # plt.annotate('Min: '+str(int((DS['rainfall']*mask.values).min()))+'. Max: '+str(int((DS['rainfall']*mask.values).max()))+'', xy=(88.8, 28.3) )
    plt.scatter(stations_x,stations_y,c=residuals_stations.values, cmap='BrBG' , vmin=vmin, vmax=vmax, edgecolors='k')
    if method_interpolation=='IDWsquared':
        plt.title(method+' residuals (IDWsquared, 4 stations, elev. x '+str(distances_selected.layering_coef.values)+')') 
    else: 
        plt.title(method+' residuals (Ordinary Kriging)') 
    plt.axis('off')
    
    plt.subplot(5,1,5)
    DS['rainfall'] = xr.DataArray(result_all[month,:,:], coords=[DS.y, DS.x])
    (DS.rainfall*mask.values).plot(cmap='nipy_spectral_r', levels=np.arange(1,10,0.2)**3 )
    border_plot()
    plt.ylabel(None)
    plt.xlabel(None)
    plt.text(0.01,0.97, 'Min: '+str(int((DS['rainfall']*mask.values).min()))+' mm. Max: '+str(int((DS['rainfall']*mask.values).max()))+' mm', transform=plt.gca().transAxes) 
    # plt.annotate('Min: '+str(int((DS['rainfall']*mask.values).min()))+' mm. Max: '+str(int((DS['rainfall']*mask.values).max()))+' mm', xy=(88.8, 28.3) )
    plt.title('Final analysis (Lin. reg + residuals) (mm)') 
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/nas8/VanishingTreasures/Paper2/Plots/precipitation_interpolation/Precipitation_lin_regression_EPSG5266_month'+str(month+1)+'_'+str(layering_coef)+'_'+method_interpolation+'.png',dpi=142.86*2) 
    plt.close('all')
    
    

