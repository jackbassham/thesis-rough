#!/bin/bash
set -e # quit if error in step

TS_RAW=04082026_0901


# echo "downloading era5..."
# python -m _01_access_data.download_wind --timestamp_raw $TS_RAW
# echo "finished downloading era5"

# echo "regridding era5..."
# python -m _02_regrid.regrid_wind --timestamp_raw $TS_RAW
# echo "finished regridding era5..."

echo "running rest of pipeline with era5..."
python -m run_pipeline --start mask_normalize --timestamp_regrid $TS_RAW --timstamp_coordinates $TS_RAW
echo "finished pipeline"