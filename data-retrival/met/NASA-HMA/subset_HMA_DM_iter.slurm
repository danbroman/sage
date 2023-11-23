#!/bin/bash
#SBATCH --account=SAGE
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=NASA-HMA
#SBATCH --output=NASA-HMA.out
#SBATCH --error=NASA-HMA.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youngjun.son@pnnl.gov

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate geospatial # conda install -c conda-forge numpy pandas xarray dask netCDF4 rioxarray

ulimit -s unlimited

year=2003

cd $HOME/projects/sage/hydro-modeling/datasets/NASA-HMA
for month in {01..12}
do
  # (default) python subset_HMA_DM_iter.py 200301 73.375,22.4375,97.8125,31.5 0.0625
  python -u subset_HMA_DM_iter.py $year$month # argment vectors: yearmonth lon_west,lat_south,lon_east,lat_north resolution(optional)
  echo Completed...$year$month
done