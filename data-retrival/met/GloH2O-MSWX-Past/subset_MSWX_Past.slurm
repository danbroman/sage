#!/bin/bash
#SBATCH --account=SAGE
#SBATCH --partition=shared
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=GloH2O-MSWX-Past
#SBATCH --output=GloH2O-MSWX-Past.out
#SBATCH --error=GloH2O-MSWX-Past.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youngjun.son@pnnl.gov

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate geospatial # conda install -c conda-forge numpy xarray dask netCDF4 (scipy)

ulimit -s unlimited

for year in {1990..2019}
do
  # (default) python subset_MSWX_Past.py 2003 73.375,22.4375,97.8125,31.5 0.0625
  python subset_MSWX_Past.py $year # argment vectors: year lon_west,lat_south,lon_east,lat_north resolution(optional)
  echo Completed...${year}
done
