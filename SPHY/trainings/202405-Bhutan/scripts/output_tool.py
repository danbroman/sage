import os
import pcraster as pcr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs
import cartopy.io.img_tiles
import cartopy.feature
import pyproj

def read_tss(path_tss, startyear, startmonth, startday):
    """
    Read SPHY timeseries outputs (path_tss: str, startyear: int, startmonth: int, startday: int)

    return pandas.DataFrame
    """
    with open(path_tss, mode = 'r') as f:
        description = f.readline().rstrip() # 1st row: timeseries description
        n_cols = f.readline().rstrip() # 2nd row: number of columns, including index
        index_name = f.readline().rstrip() # 3nd row: index (first column) name

        n_cols = int(n_cols)
        names = []
        for _ in range(1, n_cols):
            names.append(f.readline().rstrip()) # column names
    
    df_tss = pd.read_csv(path_tss, index_col = 0, skiprows = 2 + n_cols, header = None, names = names, delim_whitespace = True)
    df_tss.index = pd.Timestamp(year = startyear, month = startmonth, day = startday) + (df_tss.index - 1) * pd.Timedelta(days = 1) # daily time step
    df_tss.index.name = index_name

    return df_tss, description

def plot_tss(path_tss, id_outlet, startyear, startmonth, startday, path_obscsv = None, plot_map = False, path_outletmap = None, outletmap_epsg = None, basemap = cartopy.io.img_tiles.OSM(), basemap_level = 13, extent = 3000, figsize = (12, 5), suptitle = None, xlabel = None, ylabel = None, savefig = None):
    """
    Plot SPHY timeseries outputs

    return None
    """
    df_tss, description = read_tss(path_tss = path_tss, startyear = startyear, startmonth = startmonth, startday = startday) # read SPHY timeseries
    df_tss_sel = df_tss.loc[:, str(id_outlet)] # select an outlet ID to plot

    # read observation timeseries if avilable
    if path_obscsv is not None:
        df_obs = pd.read_csv(path_obscsv, index_col = 0, parse_dates = True)
        df_obs = df_obs.loc[df_tss.index[0]:df_tss.index[-1]].iloc[:, 0]

    # whether to plot the outlet ID location with a map
    if plot_map:
        map_outlet = pcr.readmap(path_outletmap)
        x = pcr.cellvalue(pcr.maptotal(pcr.xcoordinate(map_outlet == id_outlet)), 0)[0]
        y = pcr.cellvalue(pcr.maptotal(pcr.ycoordinate(map_outlet == id_outlet)), 0)[0]

        transformer = pyproj.Transformer.from_crs(f'EPSG:{outletmap_epsg}', 'EPSG:4326')
        lat, lon = transformer.transform(x, y)

    fig = plt.figure(figsize = figsize, layout = 'compressed')
    if suptitle is None:
        if plot_map: fig.suptitle(f'ID {id_outlet} Timeseries [lon: {round(lon, 4)} deg, lat: {round(lat, 4)} deg]')
        else: fig.suptitle(f'ID {id_outlet} Timeseries')
    else: fig.suptitle(suptitle)
    gs = matplotlib.gridspec.GridSpec(1, 3, figure = fig)

    # plot for timeseries
    if plot_map: ax1 = fig.add_subplot(gs[:2])
    else: ax1 = fig.add_subplot(gs[:])
    ax1.plot(df_tss_sel.index, df_tss_sel, 'r-', label = f'{os.path.basename(path_tss)}, ID {id_outlet}', zorder = 9)
    if path_obscsv is not None: ax1.plot(df_obs.index, df_obs, 'b-', label = os.path.basename(path_obscsv))
    if xlabel is None: ax1.set_xlabel('Date')
    else: ax1.set_xlabel(xlabel)
    if ylabel is None: ax1.set_ylabel(description)
    else: ax1.set_ylabel(ylabel)
    ax1.legend(loc = 'best'); ax1.grid()
    #fig.autofmt_xdate()

    # plot for a map with the outlet ID location
    if plot_map:
        ax2 = fig.add_subplot(gs[2], projection = basemap.crs)
        ax2.set_extent([x - extent, x + extent, y - extent, y + extent], crs = cartopy.crs.epsg(outletmap_epsg))
        ax2.add_image(basemap, basemap_level)
        fgl = ax2.gridlines(crs = cartopy.crs.PlateCarree(), linestyle = '--', alpha = 0.25, draw_labels = True)
        fgl.top_labels, fgl.right_labels = False, False,
        fgl.xlabel_style, fgl.ylabel_style = {'color': 'gray', 'size': 8}, {'color': 'gray', 'size': 8}
        ax2.scatter([x], [y], c = 'r', marker = '^', transform = cartopy.crs.epsg(outletmap_epsg))
        ax2.text(x, y - extent / 5, s = f'ID {id_outlet}\n[{int(x)}, {int(y)}]', c = 'r', transform = cartopy.crs.epsg(outletmap_epsg))

    #plt.tight_layout()
    if savefig is None: plt.show()
    else: plt.savefig(savefig, bbox_inches = 'tight')
    plt.close('all')
    
    return