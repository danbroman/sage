#!/bin/bash

domain=https://hydrology.soton.ac.uk
model=data/pgf/v3/0.25deg

for year in {1990..2016}
do
  timestep=3hourly
  for var in 'dlwrf' 'dswrf' 'pres' 'shum' 'tas' 'wind'
  do
    url=$domain/$model/$timestep/${var}_${timestep}_$year-$year.nc
    echo Target...$url
    wget -N -P ./$var/$timestep/ $url
    echo Completed...$var..$year
  done

  timestep=daily
  for var in 'prcp' 'tmax' 'tmin'
  do
    url=$domain/$model/$timestep/${var}_${timestep}_$year-$year.nc
    echo Target...$url
    wget -N -P ./$var/$timestep/ $url
    echo Completed...$var..$year
  done
done