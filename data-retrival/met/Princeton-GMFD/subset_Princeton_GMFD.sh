#!/bin/bash
# not computationally intensive - sufficient to run on a scheduler node with "screen"

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate geospatial # conda install -c conda-forge numpy xarray dask netCDF4 (scipy)

for year in {1990..2016}
do
  # (default) python subset_Princeton_GMFD.py 2003 73.375,22.4375,97.8125,31.5 0.0625
  python subset_Princeton_GMFD.py $year # argment vectors: year lon_west,lat_south,lon_east,lat_north resolution(optional)
  echo Completed...${year}
done
