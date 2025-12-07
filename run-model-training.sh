#!/bin/zsh
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# README: run-model-training.sh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PURPOSE:
#   Master script for training and evaluating all ML models with timestamped outputs
#   for version control
#
#   NOTE Includes CNN input processing '../model-training/cnn/ProcessInputs.py'
# 
# CHOOSING PARAMETERS:
#   Global variables are defined under GLOBAL CONFIG section, to 
#   configure data subset parameters. Python scripts read these via os.getenv() for reproducibility.
#   NOTE: Do not include spaces when assigining variables
#
# USAGE:
#   # Navigate to 'model-training' directory
#   cd model-training
#
#   # Initialize script
#   chmod +x run-model-training.sh
# 
#   # Run script
#   ./run-model-training.sh
#
# NOTE: TEMPORAL RANGE FOR REPLICATION
# START_YEAR=1992
# END_YEAR=2020
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL CONFIG
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define northern or southern hemisphere
export HEM="nh" # "sh" or "nh"

# Declare temporal range of input data
# NOTE should match range in input filename
export START_YEAR=1992 
export END_YEAR=2020

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN SCRIPTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get timestamp from python script
export TIMESTAMP=$(python thesis-rough/run/timestamp.py)

# Print timestamp
ok "Timestamp recorded:"
echo "  Timestamp     = $TIMESTAMP"
echo " "


# Print global variable selections
ok "Loaded global configuration:"
echo "  HEM           = $HEM"
echo "  YEARS         = $START_YEAR - $END_YEAR"
echo " "


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

exit 0