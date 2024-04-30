import pcraster as pcr
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import cartopy.crs
import cartopy.io.img_tiles
import cartopy.feature
import pyproj

# conda install -c conda-forge xarray rioxaaray dask matplotlib cartopy notebook

class clonemap:
    def __init__(self, epsg, path = None, nrow = None, ncol = None, cellsize = None, west = None, north = None):
        """
        Initialize class variables w/ EPSG
        Set clone.map if proper inputs are provided
            - Option 1. a path to read clone.map (path: str)
            - Option 2. properties to generate clone.map (nrow: int, ncol: int, cellsize: float, west: float, north: float)

        return None
        """
        self.epsg = epsg
        self.path = None
        self.setclone = None
        self.clonemap = None
        self.clone = None
        self.cellsize = None
        self.nrow, self.ncol = None, None
        self.west, self.north = None, None
        self.east, self.south = None, None
        self.xcoord, self.ycoord = None, None

        self.dem = None
        self.slope = None
        self.flowdir = None
        self.outlet = None
        self.catchment = None
        self.streamorder = None
        self.accuflux = None
        self.lat = None

        if path is not None: self.load_clonemap(path) # if a path is given to load clone.map
        elif None not in [nrow, ncol, cellsize, west, north]: self.create_clonemap(nrow = nrow, ncol = ncol, cellsize = cellsize, west = west, north = north) # if properties are given to generate clone.map
        return
    
    def set_epsg(self, epsg):
        """
        Set EPSG (epsg: int)

        return None
        """
        self.epsg = epsg
        return

    def load_clonemap(self, path):
        """
        load clone.map from a specified path (path: str)

        return None
        """
        self.path = path
        self.setclone = pcr.setclone(path)
        self.clonemap = pcr.boolean(pcr.readmap(path))
        self.update_properties()
        return
    
    def create_clonemap(self, nrow, ncol, cellsize, west, north):
        """
        Generate clone.map from provided properties (nrow: int, ncol: int, cellsize: float, west: float, north: float)

        return None
        """
        self.path = None
        self.setclone = pcr.setclone(nrow, ncol, cellsize, west, north)
        self.clonemap = self.from_numpy(array = np.ones((nrow, ncol), dtype = bool), datatype = pcr.Boolean, missing_value = -9999)
        self.update_properties()
        return
    
    def update_properties(self):
        """
        Update class variables

        return None
        """
        self.clone = pcr.clone()
        self.cellsize = self.clone.cellSize()
        self.nrow, self.ncol = self.clone.nrRows(), self.clone.nrCols()
        self.west, self.north = self.clone.west(), self.clone.north()
        self.east, self.south = self.west + self.cellsize * self.ncol, self.north - self.cellsize * self.nrow
        self.xcoord, self.ycoord = np.arange(self.west + self.cellsize / 2, self.east, self.cellsize), np.arange(self.south + self.cellsize / 2, self.north, self.cellsize)
        return
     
    def set_dem(self, field):
        """
        Set a pcraster field of DEM to process
            - Option 1. a path to a pcraster file (field: str)
            - Option 2. a pcraster field (field: pcraster.Field)

        return pcraster.Field
        """
        if isinstance(field, str):  # Option 1. a path to a pcraster file
            self.dem = pcr.readmap(field)
        elif isinstance(field, pcr.Field): # Option 2. a pcraster field
            self.dem = field
        return self.dem
    
    def fill_dem(self, outflowdepth = 1e31, corevolume = 1e31, corearea = 1e31, catchmentprecipitation = 1e31, lddin = False, replace = True):
        """
        Calculate pcrasters of depression-filled DEM: a wrapper for pcraster.lddcreatedem
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_lddcreatedem.html for pcraster.lddcreatedem function arguments

        return pcraster.Field
        """
        if self.dem is None: return None
        if lddin: pcr.setglobaloption('lddin')
        field = pcr.lddcreatedem(self.dem, outflowdepth, corevolume, corearea, catchmentprecipitation)
        if replace: self.dem = field
        return field
    
    def calculate_slope(self):
        """
        Calculate a pcraster of slope: a wrapper for pcraster.slope
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_slope.html for pcraster.slope function arguments

        return pcraster.Field
        """
        if self.dem is None: return None
        self.slope = pcr.slope(self.dem)
        return self.slope
    
    def calculate_flowdir(self, outflowdepth = 1e31, corevolume = 1e31, corearea = 1e31, catchmentprecipitation = 1e31, lddin = False):
        """
        Calculate pcrasters of flow direction: a wrapper for pcraster.lddcreate
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_lddcreate.html for pcraster.lddcreate function arguments
        
        return pcraster.Field
        """
        if self.dem is None: return None
        if lddin: pcr.setglobaloption('lddin')
        self.flowdir = pcr.lddcreate(self.dem, outflowdepth, corevolume, corearea, catchmentprecipitation)
        return self.flowdir
    
    def create_outlet(self, outlets, epsg = None):
        """
        Create a pcraster field of outlets and a list of outlets on a clonemap EPSG (outlets: list/numpy.ndarray, e.g., [[lat1, lon1], [lat2, lon2], ...] for EPSG4326, epsg: int)

        return pcraster.Field, numpy.ndarray
        """
        if epsg is None: epsg = self.epsg
        if self.epsg == epsg: outlets_epsg = outlets
        else:
            transformer = pyproj.Transformer.from_crs(f'EPSG:{epsg}', f'EPSG:{self.epsg}')
            outlets_epsg = transformer.transform(np.array(outlets)[:, 0], np.array(outlets)[:, 1])
            outlets_epsg = np.array(list(zip(*outlets_epsg)))
        
        da = self.to_xarray()
        da.loc[:] = 0
        for i, p in enumerate(outlets_epsg):
            da_sel = da.sel(x = p[0], y = p[1], method = 'nearest')
            da.loc[{'x': da_sel['x'], 'y': da_sel['y']}] = 1 + i
        self.outlet = self.from_xarray(dataarray = da, datatype = pcr.Nominal, missing_value = 0)
        return self.outlet, outlets_epsg
    
    def create_catchment(self):
        """
        Create a pcraster of catchment: a wrapper for pcraster.catchment
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_catchment.html

        return pcraster.Field
        """
        if self.flowdir is None or self.outlet is None: return None
        self.catchment = pcr.catchment(self.flowdir, self.outlet)
        return self.catchment
    
    def create_subcatchment(self):
        """
        Create a pcraster of catchment: a wrapper for pcraster.subcatchment
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_subcatchment.html

        return pcraster.Field
        """
        if self.flowdir is None or self.outlet is None: return None
        self.catchment = pcr.subcatchment(self.flowdir, self.outlet)
        return self.catchment
    
    def create_streamorder(self):
        """
        Create a pcraster of streamorder: a wrapper for pcraster.streamorder
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_streamorder.html

        return pcraster.Field
        """
        if self.flowdir is None: return None
        self.streamorder = pcr.streamorder(self.flowdir)
        return self.streamorder
    
    def create_accuflux(self, material = 1):
        """
        Create a pcraster of accumulated flux: a wrapper for pcraster.accuflux
        See https://pcraster.geo.uu.nl/pcraster/4.4.1/documentation/pcraster_manual/sphinx/op_accuflux.html for pcraster.lddcreatedem function arguments

        return pcraster.Field
        """
        if self.flowdir is None: return None
        self.accuflux = pcr.accuflux(self.flowdir, material)
        return self.accuflux
    
    def create_latitude(self, field_mv = None):
        """
        Create a pcraster of latitude
        
        return pcraster.Field
        """
        if self.epsg == 4326: _, latgrid = np.meshgrid(self.xcoord, self.ycoord)
        else:
            transformer = pyproj.Transformer.from_crs(f'EPSG:{self.epsg}', 'EPSG:4326')
            xgrid, ygrid = np.meshgrid(self.xcoord, self.ycoord)
            latgrid, _, _ = transformer.transform(xgrid, ygrid, np.zeros_like(xgrid)) 

        if field_mv is not None: latgrid[np.isnan(self.to_numpy(field_mv))] = -9999
        self.lat = self.from_numpy(array = latgrid, datatype = pcr.Scalar, missing_value = -9999)
        return self.lat
    
    def extract_mask(self, field):
        """
        Extract a pcraster field of missing value
            - Option 1. a path to a pcraster file (field: str)
            - Option 2. a pcraster field (field: pcraster.Field)

        return pcraster.Field
        """
        return self.from_numpy(np.isnan(self.to_numpy(field)), datatype = pcr.Boolean, missing_value = -1)

    def from_numpy(self, array, datatype, missing_value):
        """
        Convert a numpy array to a pcraster field (array: numpy.ndarray, datatype: pcraster.Boolean/Nominal/Ordinal/Scalar/Directional/Ldd, missing_values: int/float?)

        return pcraster.Field
        """
        field = pcr.numpy2pcr(dataType = datatype, array = np.flipud(array).copy(), mv = missing_value) # generate an error without .copy()
        return field
    
    def from_xarray(self, dataarray, datatype, missing_value, xdim = 'x', ydim = 'y', tdim = 'time'):
        """
        Convert a xarray dataarray to a pcraster field (array: xarray.DataArray, datatype: pcraster.Boolean/Nominal/Ordinal/Scalar/Directional/Ldd, missing_values: int/float?, xdim: str, ydim: str, tdim: str)

        return pcraster.Field
        """
        if tdim in dataarray.dims: # iterate over time if time is in dimensions
            da = dataarray.transpose(tdim, ydim, xdim).sortby([tdim, ydim, xdim])
            field = []
            for t in da[tdim]:
                field.append(self.from_numpy(array = da.sel({tdim: t}).values, datatype = datatype, missing_value = missing_value))
        else:
            da = dataarray.transpose(ydim, xdim).sortby([ydim, xdim])
            field = self.from_numpy(array = da.values, datatype = datatype, missing_value = missing_value)
        return field
    
    def to_numpy(self, field = None):
        """
        Convert a pcraster field(s) to a numpy array
            - Option 1. clone.map (field: None)
            - Option 2. a path to a pcraster file (field: str)
            - Option 3. a pcraster field (field: pcraster.Field)
            - Option 4. a list of pcrastser fields/paths, e.g., time-series (field: list/numpy.ndarray)

        return numpy.ndarray
        """
        data = None
        if field is None: # Option 1. clone.map
            data = pcr.pcr2numpy(map = self.clonemap, mv = np.nan)
            data = np.flipud(data)
        elif isinstance(field, str):  # Option 2. a path to a pcraster file
            data = pcr.pcr2numpy(map = pcr.readmap(field), mv = np.nan)
            data = np.flipud(data)
        elif isinstance(field, pcr.Field): # Option 3. a pcraster field
            data = pcr.pcr2numpy(map = field, mv = np.nan)
            data = np.flipud(data)
        elif isinstance(field, (list, np.ndarray)): # Option 4. a list of pcrastser fields/paths, e.g., time-series
            if isinstance(field[0], pcr.Field): data = np.array([pcr.pcr2numpy(map = f, mv = np.nan) for f in field])
            elif isinstance(field[0], str): data = np.array([pcr.pcr2numpy(map = pcr.readmap(f), mv = np.nan) for f in field])
            data = np.flip(data, axis = 1)
        else: return None
        return data
    
    def to_xarray(self, field = None, tcoord = None, varname = None):
        """
        Convert a pcraster field(s) to a xarray dataarray
            - Option 1. clone.map (field: None, varname [optional]: str)
            - Option 2. a path to a pcraster file (field: str, varname [optional]: str)
            - Option 3. a pcraster field (field: pcraster.Field, varname [optional]: str)
            - Option 4. a list of pcrastser fields, e.g., time-series (field: list/numpy.ndarray, tcoord: list/numpy.ndarray, varname [optional]: str)

        return xarray.DataArray
        """
        data = self.to_numpy(field = field)
        dims, coords = ['y', 'x'], {'y': self.ycoord, 'x': self.xcoord}
        if isinstance(field, (list, np.ndarray)): # Option 4. a list of pcrastser fields, e.g., time-series
            dims = ['time'] + dims
            if tcoord is None: coords['time'] = np.arange(len(field))
            else: coords['time'] = tcoord
        
        da = xr.DataArray(data = data, dims = dims, coords = coords)
        if varname is not None: da.name = varname
        return da
    
    def read_pcraster(self, path):
        """
        read clone.map from a specified path (path: str)

        return pcraster.Field
        """
        field = pcr.readmap(path)
        return field
    
    def read_netcdf(self, path, varname, epsg, datatype, xdim = 'x', ydim = 'y', tdim = 'time', tcoord = None):
        """
        read a netcdf file(s) as a pcraster.Field from a specified path (path: str/list/numpy.ndarray, varname: str, epsg: int, datatype: pcraster.Boolean/Nominal/Ordinal/Scalar/Directional/Ldd, missing_values: int/float?, xdim: str, ydim: str, tdim: str, tcoord: list/numpy.ndarray)

        return pcraster.Field or a list of pcraster.Fields
        """
        da = xr.open_mfdataset(path)[varname]
        if tdim in da.dims:
            if tcoord is not None: da = da.sel({tdim: tcoord})
            da = da.rename({xdim: 'x', ydim: 'y', tdim: 'time'})
        else: da = da.rename({xdim: 'x', ydim: 'y'})
        da = da.rio.write_crs(epsg, inplace = True).rio.write_coordinate_system(inplace = True)
        if self.epsg != epsg:
            da = da.rio.reproject_match(self.to_xarray().rio.write_crs(self.epsg, inplace = True).rio.write_coordinate_system(inplace = True))
            da = da.assign_coords({'x': self.xcoord, 'y': self.ycoord})
        field = self.from_xarray(dataarray = da, datatype = datatype, missing_value = da.rio.nodata)
        return field
    
    def read_raster(self, path, epsg, datatype, band = None):
        """
        read a raster file(s) as a pcraster.Field from a specified path (path: str/list/numpy.ndarray, epsg: int, datatype: pcraster.Boolean/Nominal/Ordinal/Scalar/Directional/Ldd, missing_values: int/float?, band: int)

        return pcraster.Field or a list of pcraster.Fields
        """
        if isinstance(path, str): da = rioxarray.open_rasterio(path)
        elif isinstance(path, (list, np.ndarray)): da = xr.concat([rioxarray.open_rasterio(p) for p in path], dim = 'time')
        if 'band' in da.dims:
            if band is not None: da = da.sel(band = band)
        da = da.rio.write_crs(epsg, inplace = True).rio.write_coordinate_system(inplace = True)
        if self.epsg != epsg:
            da = da.rio.reproject_match(self.to_xarray().rio.write_crs(self.epsg, inplace = True).rio.write_coordinate_system(inplace = True))
            da = da.assign_coords({'x': self.xcoord, 'y': self.ycoord})
        field = self.from_xarray(dataarray = da, datatype = datatype, missing_value = da.rio.nodata)
        return field

    def write_pcraster(self, path, field = None):
        """
        write a pcraster file(s) to a specified path(s)
            - Option 1. clone.map (path: str, field: None)
            - Option 2. a path to a pcraster file (path: str, field: str)
            - Option 3. a pcraster field (path: str, field: pcraster.Field)
            - Option 4. a list of pcrastser fields, e.g., time-series (path: list/numpy.ndarray, field: list/numpy.ndarray)

        return None
        """
        if field is None: # Option 1. clone.map
            pcr.report(self.clonemap, path)
            print(f'write_pcraster - clone.map: written to {path}.')
        elif isinstance(field, (str, pcr.Field)): # Option 2. a path to a pcraster file and Option 3. a pcraster field
            pcr.report(field, path)
            print(f'write_pcraster: written to {path}.')
        elif isinstance(field, (list, np.ndarray)): # Option 4. a list of pcrastser fields, e.g., time-series
            for i, f in enumerate(field):
                pcr.report(f, path[i])
                print(f'write_pcraster: written to {path[i]} - #{1 + i}/{len(field)}.')
        return
    
    def write_netcdf(self, path, field = None, tcoord = None, varname = None, epsg = None):
        """
        write a netcdf file to a specified path
            - Option 1. clone.map (path: str, field: None, varname [optional]: str, epsg [optional]: int)
            - Option 2. a path to a pcraster file (path: str, field: str, varname [optional]: str, epsg [optional]: int)
            - Option 3. a pcraster field (path: str, field: pcraster.Field, varname [optional]: str, epsg [optional]: int)
            - Option 4. a list of pcrastser fields, e.g., time-series (path: str, field: list/numpy.ndarray, tcoord: list/numpy.ndarray, varname [optional]: str, epsg [optional]: int)

        return None
        """
        da = self.to_xarray(field = field, tcoord = tcoord, varname = varname)
        da = da.rio.write_crs(self.epsg, inplace = True).rio.write_coordinate_system(inplace = True)
        if epsg is not None and self.epsg != epsg: da = da.rio.reproject(f'EPSG:{epsg}')
        da.to_netcdf(path)
        print(f'write_netcdf: written to {path}.')
        return
            
    def write_raster(self, path, field = None, tcoord = None, varname = None, epsg = None):
        """
        write a raster (GeoTiff) file(s) to a specified path(s)
            - Option 1. clone.map (path: str, field: None, varname [optional]: str, epsg [optional]: int)
            - Option 2. a path to a pcraster file (path: str, field: str, varname [optional]: str, epsg [optional]: int)
            - Option 3. a pcraster field (path: str, field: pcraster.Field, varname [optional]: str, epsg [optional]: int)
            - Option 4. a list of pcrastser fields, e.g., time-series (path: list/numpy.ndarray, field: list/numpy.ndarray, tcoord: list/numpy.ndarray, varname [optional]: str, epsg [optional]: int)

        return None
        """
        da = self.to_xarray(field = field, tcoord = tcoord, varname = varname)
        da = da.rio.write_crs(self.epsg, inplace = True).rio.write_coordinate_system(inplace = True)
        if epsg is not None and self.epsg != epsg: da = da.rio.reproject(f'EPSG:{epsg}')
        if isinstance(field, (list, np.ndarray)): # Option 4. a list of pcrastser fields, e.g., time-series
            for i, t in enumerate(da['time']):
                da.sel(time = t).rio.to_raster(path[i])
                print(f'write_raster: written to {path[i]} - #{1 + i}/{len(field)}.')
        else:
            da.rio.to_raster(path)
            print(f'write_raster: written to {path}.')
        return
    
    def plot_cartopy(self, field, idx_plot = None, shapes = None, figsize = None, basemap = cartopy.io.img_tiles.OSM(), basemap_level = 10, extent = None, vmin = None, vmax = None, cmap = 'turbo', alpha = 0.7, title = None, clabel = None, savefig = None):
        """
        Plot a pcraster field(s), using matplotlib and cartopy
            - Option 1. clone.map (field: None)
            - Option 2. a path to a pcraster file (field: str)
            - Option 3. a pcraster field (field: pcraster.Field)
            - Option 4. one instance from a list of pcrastser fields, e.g., time-series (field: list/numpy.ndarray, time: datetime.datetime/pandas.Timestamp)

        return
        """
        da = self.to_xarray(field = field, varname = clabel)
        if idx_plot is not None: da = da.isel(time = idx_plot)
        
        fig, ax = plt.subplots(figsize = figsize, subplot_kw = {'projection': basemap.crs})
        ax.add_image(basemap, basemap_level)
        if extent is None: ax.set_extent([self.west, self.east, self.south, self.north], crs = cartopy.crs.epsg(self.epsg))
        else: ax.set_extent(extent, crs = cartopy.crs.PlateCarree())
        fgl = ax.gridlines(crs = cartopy.crs.PlateCarree(), linestyle = '--', alpha = 0.25, draw_labels = True)
        fgl.top_labels, fgl.right_labels = False, False,
        fgl.xlabel_style, fgl.ylabel_style = {'color': 'gray', 'size': 8}, {'color': 'gray', 'size': 8}
        fda = da.plot(ax = ax, transform = cartopy.crs.epsg(self.epsg), vmin = vmin, vmax = vmax, cmap = cmap, alpha = alpha)
        if shapes is not None:
            for s in shapes: ax.add_feature(s)
        if title is not None: ax.set_title(title)
        fig.tight_layout()
        if savefig is not None: plt.savefig(savefig, bbox_inches = 'tight')
        else: plt.show()
        plt.close('all')
        return