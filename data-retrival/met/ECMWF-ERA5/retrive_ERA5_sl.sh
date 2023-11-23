#!/bin/bash
# not computationally intensive - sufficient to run on a scheduler node with "screen"

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate cdsapi # conda install -c conda-forge cdsapi eccodes

for year in {1990..2019}
do
  # (default) python retrive_ERA5_sl.py 2003 73.375,22.4375,97.8125,31.5
  python retrive_ERA5_sl.py $year # argment vectors: year lon_west,lat_south,lon_east,lat_north
done
