#!/bin/bash
# not computationally intensive - sufficient to run on a scheduler node with "screen"

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate geospatial # conda install -c conda-forge numpy xarray dask netCDF4 (scipy)

for year in {1981..2011..10}
do
  # (default) python concat_GSWP3_W5E5.py 2001 73.375,22.4375,97.8125,31.5 0.0625
  python concat_GSWP3_W5E5.py $year # argument vectors: year(10-year interval) lon_west,lat_south,lon_east,lat_north(optional) resolution(optional)
  echo Completed...${year}s
done
