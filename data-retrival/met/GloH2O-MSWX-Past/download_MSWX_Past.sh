#!/bin/bash
# wget https://downloads.rclone.org/rclone-current-linux-amd64.zip
# unzip rclone-current-linux-amd64.zip

config=GoogleDrive_GT # require "rclone" configuration setup
domain=MSWX_V100
model=Past

for year in {1990..2019}
do
  timestep=3hourly
  for var in 'LWd' 'P' 'Pres' 'RelHum' 'SpecHum' 'SWd' 'Temp' 'Wind'
  do
    folder=$config:/$domain/$model/$var/$timestep/
    echo Target...$folder/${year}*.nc
    mkdir -p ./$var/$timestep
    ~/bin/rclone copy --progress --drive-shared-with-me --include ${year}*.nc $folder ./$var/$timestep/
    echo Completed...${var}..${year}
  done

  timestep=Daily
  for var in 'LWd' 'P' 'Pres' 'RelHum' 'SpecHum' 'SWd' 'Temp' 'Wind' 'Tmax' 'Tmin'
  do
    folder=$config:/$domain/$model/$var/$timestep/
    echo Target...$folder/${year}*.nc
    mkdir -p ./$var/$timestep
    ~/bin/rclone copy --progress --drive-shared-with-me --include ${year}*.nc $folder ./$var/$timestep/
    echo Completed...${var}..${year}
  done
done