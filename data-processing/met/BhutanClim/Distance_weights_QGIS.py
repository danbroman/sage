#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:28:05 2021

@author: fabianl
"""
import numpy as np
import xarray as xr
import pandas as pd
import grass.script as gs
import pyproj 

from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.modules.shortcuts import raster as r

# import os
# os.chdir("/nas8/VanishingTreasures/grassdata/Bhutan/Bhutan")

''' open this script only with the GRASS shell in QGIS GRASS Tools and open python/spyder here. 
This script will not work when python/spyder is opened in the Linux terminal'''





stations =  pd.read_csv('/nas8/VanishingTreasures/data/Corrected_metadata.csv',  decimal='.', index_col=0) 

WGS84=pyproj.Proj("EPSG:4326") # LatLon with WGS84 
BhutanLambert=pyproj.Proj("EPSG:5266")  # projection of Bhutan
transformer = pyproj.Transformer.from_proj(WGS84,BhutanLambert)
stations_x,stations_y = transformer.transform(np.array(stations.latitude),np.array(stations.longitude))


layers=[75,100] #[1,5,10,15,20,30,40,50]
for k in range(len(layers)):
    layering_coef = layers[k]
    inputs = range(len(stations))
    # for real topo
    for i in inputs: # for every station
    
        r.walk(flags = 'k' , elevation="topo_grass@Bhutan_5266", friction="ones_grass@Bhutan_5266", start_coordinates=(stations_x[i], stations_y[i]), 
               walk_coeff=(1, layering_coef, -layering_coef, -layering_coef), output="distances", max_cost=0, overwrite=True)    
        
        r.out_gdal(flags = 'f', input="distances@Bhutan_5266",  format="netCDF",  type="Float32",  output="/nas8/VanishingTreasures/Paper2/data/distances/"+str(layering_coef)+"/distances_statnr"+str(i).zfill(3)+"_filled_0.nc", overwrite=True)
    
    
    ''' Höhenunterschied zwischen Topografie und Station berücksichtigen!!!!!!!!!!  '''
    filling_coef = [0,3,5,7,9,11,15,21,26,31,41,51,71,101]
    for j in range(1,len(filling_coef)):
        for i in range(len(inputs)): # for every station
        
            r.walk(flags = 'k', elevation="topo_grass_"+str(filling_coef[j])+"@Bhutan_5266", friction="ones_grass@Bhutan_5266", start_coordinates=(stations_x[i], stations_y[i]), 
                   walk_coeff=(1, layering_coef, -layering_coef, -layering_coef), output="distances", max_cost=0, overwrite=True)        
            
            r.out_gdal(flags = 'f', input="distances@Bhutan_5266",  format="netCDF",  type="Float32",  output="/nas8/VanishingTreasures/Paper2/data/distances/"+str(layering_coef)+"/distances_statnr"+str(i).zfill(3)+"_filled_"+str(filling_coef[j])+"_uncorrected.nc", overwrite=True)


#%%  correct the raw output distances: 
