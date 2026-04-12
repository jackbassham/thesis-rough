#!/bin/bash
set -e # quit if error in step

TS=04082026_0901

echo "downloading era5..."
python -m _01_access_data.download_wind --timestamp_out $TS
echo "finished downloading era5"

echo "regridding era5..."
python -m _02_regrid.regrid_wind --timestamp_out $TS
echo "finished regridding era5..."

echo "running rest of pipeline with era5..."
python -m run_pipeline --start mask_normalize --timestamp_out $TS
echo "finished pipeline"