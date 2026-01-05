#!/bin/zsh
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# README: run-model-training.sh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PURPOSE:
#   Master script for downloading, processing data, and 
#   training and evaluating all ML models with timestamped outputs
#   for version control
#
# 
# CHOOSING PARAMETERS:
#   Global variables are defined under GLOBAL CONFIG section, to 
#   configure data subset parameters. Python scripts read these via os.getenv() for reproducibility.
#   NOTE: Do not include spaces when assigining variables
#
# USAGE:
#   # Navigate to 'thesis-rough' repository directory
#   cd
#
#   # Initialize script
#   chmod +x run-model-training.sh
# 
#   # Run script
#   ./run-model-training.sh
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
# NOTE: MIN 1989 (HOFFMAN), 1989-1991 faulty JRA55 data on Mazloff Server
# TODO: Download JRA55 data from source
# START_YEAR=1992
# END_YEAR=2020
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL CONFIG
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Enter NASA Earthdata Login Credentials HERE
# (For NSIDC Motion and Concentration Datasets)
export USER="jbassham"
export PASS="guJdib-huczi6-jimsuh"

# Define northern or southern hemisphere
export HEM="nh" # "sh" or "nh"

# Define latitude longitude bounds (see above for replication)
# NOTE: Limits are in quotes, separated by comma
export LAT_LIMITS="60, 90" # "lower (Southern), upper (Northern)", (coverage 29.7N to 90N or -90S to -37S)
export LON_LIMITS="-180, 180"  # "Western, Eastern", (coverage -180 W to 180E)

# Define grid resolution
export RESOLUTION=25 # km

# Define temporal range (see above for replication)
export START_YEAR=1992 # minimum 1989
export END_YEAR=2020

# Define northern or southern hemisphere
export HEM="nh" # "sh" or "nh"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN SCRIPTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get timestamp from python script
export TIMESTAMP=$(python -m helpers.timestamp.py)

# Print timestamp
ok "Timestamp recorded:"
echo "  Timestamp     = $TIMESTAMP"
echo " "


# Print global variable selections
ok "Global variables:"
echo "  HEM           = $HEM"
echo "  YEARS         = $START_YEAR - $END_YEAR"
echo " "

# For this master shell script, all input/ output timestamp versions will be the same
# For individual shell scripts (model training, etc, enter appropriate timestamp versions for input data)

export TIMESTAMP_IN=$TIMESTAMP
export TIMESTAMP_OUT=$TIMESTAMP

echo "1. DOWNLOADING RAW DATA"

echo "Starting 1-download/download_concentration_nimbus7.py..."
if ! python -m 1-download.download_concentration_nimbus7.py; then 
    echo "ERROR: Failed to run 1-download/download_concentration_nimbus7.py"
    exit 1
fi

echo "Finished 1-download/download_concentration_nimbus7.py,"
echo "Starting 1-download/download_wind_jra55.py..."
if ! python -m 1-download.download_wind_jra55.py; then 
    echo "ERROR: Failed to run 1-download/download_wind_jra55.py"
    exit 1
fi

echo "Finished 1-download/download_wind_jra55.py,"
echo "Starting 1-download/download_motion_pp.py..."
if ! python -m 1-download.download_motion_pp.py; then 
    echo "ERROR: Failed to run 1-download/download_motion_pp.py"
    exit 1
fi

echo "2. REGRIDDING DATA"

echo "Starting 2-regrid/regrid_concentration_nimbus7.py..."
if ! python -m 2-regrid.regrid_concentration_nimbus7.py; then 
    echo "ERROR: Failed to run regrid_concentration_nimbus7.py"
    exit 1
fi

echo "Finished 2-regrid/regrid_concentration_nimbus7.py," 
echo "Starting 2-regrid/regrid_wind_jra55.py..."
if ! python -m 2-regrid.regrid_wind_jra55.py; then 
    echo "ERROR: Failed to run 2-regrid/regrid_wind_jra55.py"
    exit 1
fi

echo "Finished 2-regrid/regrid_wind_jra55.py," 
echo "Starting 2-regrid/regrid_motion_pp.py..."
if ! python -m 2-regrid.regrid_motion_pp.py; then 
    echo "ERROR: Failed to run 2-regrid/regrid_motion_pp.py"
    exit 1
fi


echo "3. MASK & NORMALIZE DATA"

echo "Starting 3-mask-normalize/mask_normalize_inputs.py..."
if ! python -m 3-mask-normalize.mask_normalize.py; then 
    echo "ERROR: Failed to run 3-mask-normalize/mask_normalize_inputs.py"
    exit 1
fi

echo "4. PROCESS MODEL INPUTS" 

echo "Starting 4-process-inputs/mask_normalize_inputs.py..."
if ! python -m 3-mask-normalize.mask_normalize.py; then 
    echo "ERROR: Failed to run 3-mask-normalize/mask_normalize_inputs.py"
    exit 1
fi

echo "Starting ML Model Training"
echo " "

# echo "1. Starting Persistance"

echo "2. Training Linear Regression"
echo "Running '5-lr/lr_cf.py'"
echo " "
if ! python 5-lr/lr_cf.py; then
    echo "ERROR: Failed to run lr_cf.py"
    exit 1
fi

echo "3. Training Weighted Linear Regression"
echo "Running '6-wlr/wlr_cf.py'"
echo " "
if ! python 6-wlr/wlr_cf.py; then
    echo "ERROR: Failed to run wlr_cf.py"
    exit 1
fi

echo "4. Training CNN"
echo "Running '7-cnn/cnn_cf.py'"
echo " "
if ! python 7-cnn/cnn_cf.py; then
    echo "ERROR: Failed to run cnn_cf.py"
    exit 1
fi

echo "5. Training Weigghted CNN"
echo "Running '8-wcnn/wcnn_cf.py'"
echo " "
if ! python 8-wcnn/wcnn_cf.py; then
    echo "ERROR: Failed to run wcnn_cf.py"
    exit 1
fi

echo "Model Training Complete!"
echo " "

echo "All outputs saved with timestamp: $TIMESTAMP"
echo " "

exit 0