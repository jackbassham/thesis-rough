#!/bin/zsh
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# README: run-data-processing.sh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PURPOSE:
#   Master script for downloading, regriding, masking, and normalizing data
#   for ML models
#   NOTE: CNN needs additional at '../model-training/cnn/ProcessInputs.py'
# 
# CHOOSING PARAMETERS:
#   Change global variables under GLOBAL CONFIG section to 
#   configure data subset parameters
#   NOTE: Do not include spaces when assigining variables
#
# USAGE:
#   # Initialize script
#   chmod +x run-data-processing.sh
# 
#   # Run script
#   ./run-data-processing.sh
#
#
# NOTE: LATITUDE LONGITUDE BOUNDS FOR REPLICATION
#
# SOUTHERN OCEAN BOUNDS
# LAT_LIMITS="-80, -62" # Enter South to North (coverage 29.7N to 90N or -90S to -37S)
# LON_LIMITS="-180, 180" # Enter West to East (coverage -180 W to 180E)
#
# ARCTIC BOUNDS
# LAT_LIMITS="60, 90" 
# LON_LIMITS="-180, 180"
#
# NOTE: TEMPORAL RANGE FOR REPLICATION
# START_YEAR=1992
# END_YEAR=2020
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL CONFIG
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define northern or southern hemisphere
export HEM="nh" # "nh" or "sh"

# Define latitude longitude bounds
# NOTE: Limits are in quotes, separated by comma
export LAT_LIMITS="60, 90" # "lower (Northern), upper (Northern)", (coverage 29.7N to 90N or -90S to -37S)
export LON_LIMITS="-180, 180"  # "Western, Eastern", (coverage -180 W to 180E)

# Define grid resolution
export RESOLUTION=25 # km

# Define temporal range
export START_YEAR=1992 # minimum 1989
export END_YEAR=2020

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN SCRIPTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo "1. Starting Downloads"

echo "Starting download_concentration_nimbus7.py..."
if ! python download_concentration_nimbus7.py; then 
    echo "ERROR: Failed to run download_concentration_nimbus7.py"
    exit 1
fi

echo "Finished download_concentration_nimbus7.py, Starting download_wind_jra55.py..."
if ! python download_wind_jra55.py; then 
    echo "ERROR: Failed to run download_wind_jra55.py"
    exit 1
fi

echo "Finished download_wind_jra55.py, Starting download_motion_pp.py..."
if ! python download_motion_pp.py; then 
    echo "ERROR: Failed to run download_motion_pp.py"
    exit 1
fi

echo "2. Starting Regrid"

echo "Starting regrid_concentration_nimbus7.py..."
if ! python regrid_concentration_nimbus7.py; then 
    echo "ERROR: Failed to run regrid_concentration_nimbus7.py"
    exit 1
fi

echo "Finished regrid_concentration_nimbus7.py, Starting regrid_wind_jra55.py..."
if ! python regrid_wind_jra55.py; then 
    echo "ERROR: Failed to run regrid_wind_jra55.py"
    exit 1
fi

echo "Finished regrid_wind_jra55.py, Starting regrid_motion_pp.py..."
if ! python regrid_motion_pp.py; then 
    echo "ERROR: Failed to run regrid_motion_pp.py"
    exit 1
fi

echo "3. Starting Mask, Normalization"

echo "Starting mask_normalize_inputs.py..."
if ! python mask_normalize_inputs.py; then 
    echo "ERROR: Failed to run mask_normalize_inputs.py"
    exit 1
fi
