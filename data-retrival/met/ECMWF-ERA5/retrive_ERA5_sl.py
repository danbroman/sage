import sys, os, subprocess
import cdsapi 

''' 
    ERA5 hourly data on single levels from 1940 to present:
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
    https://doi.org/10.24381/cds.e2161bac
    
    Requried package installations: conda install -c conda-forge cdsapi eccodes


'''
c = cdsapi.Client() # require a CDS credential, see https://cds.climate.copernicus.eu/api-how-to

def retrive_yb(year, bounds, gribfile):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'snowfall', 'surface_net_solar_radiation',
                'surface_pressure', 'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards',
                'total_precipitation',
            ],
            'year': '{}'.format(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                bounds[3], bounds[0], bounds[1], bounds[2],
            ],
            'format': 'grib',
        },
        gribfile)
    
    return

if __name__ == '__main__':
    year = 2003
    bounds = [73.375, 22.4375, 97.8125, 31.5]

    if len(sys.argv) > 1: year = int(sys.argv[1])
    if len(sys.argv) > 2: bounds = [float(v) for v in sys.argv[2].split(',')]
    
    gribfile = 'ERA5_sl_subset_{}.grib'.format(year) # output grib filename
    os.makedirs('raw', exist_ok = True)
    retrive_yb(year, bounds, os.path.join('raw', gribfile))
    subprocess.run(['grib_to_netcdf', '-o', gribfile.replace('.grib', '.nc'), os.path.join('raw', gribfile)]) # convert grib to netcdf