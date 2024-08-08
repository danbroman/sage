# QGIS Python commands for developing snow groups

# aspect from DEM


# bin DEM into 500m elevation bands
processing.run("native:reclassifybytable", {'INPUT_RASTER':'/Users/brom374/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/FY2023/SAGE/Calc/snow_glacier_calib/dem_bhutan.tif','RASTER_BAND':1,'TABLE':['0','500','1','500','1000','2','1000','1500','3','1500','2000','4','2000','2500','5','2500','3000','6','3000','3500','7','3500','4000','8','4000','4500','9','4500','5000','10','5000','5500','11','5500','6000','12','6000','6500','13','6500','7000','14'],'NO_DATA':-9999,'RANGE_BOUNDARIES':1,'NODATA_FOR_MISSING':False,'DATA_TYPE':1,'OUTPUT':'TEMPORARY_OUTPUT'})

# bin aspect into 8 direction quadrants
processing.run("native:reclassifybytable", {'INPUT_RASTER':'/Users/brom374/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/FY2023/SAGE/Calc/snow_glacier_calib/aspect_bhutan.tif','RASTER_BAND':1,'TABLE':['22.5','67.5','2','67.5','112.5','3','112.5','157.5','4','157.5','202.5','5','202.5','247.5','6','247.5','292.5','7','292.5','337.5','8','337.5','360','1','0','22.5','1'],'NO_DATA':-9999,'RANGE_BOUNDARIES':1,'NODATA_FOR_MISSING':False,'DATA_TYPE':1,'OUTPUT':'TEMPORARY_OUTPUT'})

# bin vegetation

# create unique groups based on bins



