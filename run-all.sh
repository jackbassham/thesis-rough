#!/bin/zsh
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# README: run-all.sh
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
#
#   # Initialize script
#   chmod +x run-all.sh
# 
#   # Run script
#   ./run-all.sh
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZE LOG FOR SHELL SCRIPT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOGDIR="logs"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/run_$(date +'%Y%m%d_%H%M%S').log"

# Redirect ALL output to log file (and still show it in terminal)
exec > >(tee -a "$LOGFILE") 2>&1

echo "===== PIPELINE START ====="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "=========================="

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ACTIVATE CONDA ENVIRONMENT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate seaice

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN SCRIPTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "1. DOWNLOAD RAW DATA"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo "Starting download_concentration_nimbus7.py..."
if ! python -m 1-download.download_concentration_nimbus7.py; then 
    echo "ERROR: Failed to run 1-download/download_concentration_nimbus7.py"
    exit 1
fi

echo "Finished download_concentration_nimbus7.py,"
echo "Starting download_wind_jra55.py..."
if ! python -m 1-download.download_wind_jra55.py; then 
    echo "ERROR: Failed to run 1-download/download_wind_jra55.py"
    exit 1
fi

echo "Finished download_wind_jra55.py,"
echo "Starting download_motion_pp.py..."
if ! python -m 1-download.download_motion_pp.py; then 
    echo "ERROR: Failed to run 1-download/download_motion_pp.py"
    exit 1
fi

echo "Finished download_motion_pp.py"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "2. REGRID DATA"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo "Starting regrid_concentration_nimbus7.py..."
if ! python -m 2-regrid.regrid_concentration_nimbus7.py; then 
    echo "ERROR: Failed to run regrid_concentration_nimbus7.py"
    exit 1
fi

echo "Finished regrid_concentration_nimbus7.py," 
echo "Starting regrid_wind_jra55.py..."
if ! python -m 2-regrid.regrid_wind_jra55.py; then 
    echo "ERROR: Failed to run 2-regrid/regrid_wind_jra55.py"
    exit 1
fi

echo "Finished regrid_wind_jra55.py," 
echo "Starting regrid_motion_pp.py..."
if ! python -m 2-regrid.regrid_motion_pp.py; then 
    echo "ERROR: Failed to run 2-regrid/regrid_motion_pp.py"
    exit 1
fi

echo "Finished regrid_motion_pp.py" 
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "3. MASK & NORMALIZE DATA"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo "Starting mask_normalize_inputs.py..."
if ! python -m 3-mask-normalize.mask_normalize.py; then 
    echo "ERROR: Failed to run 3-mask-normalize/mask_normalize_inputs.py"
    exit 1
fi

echo "Finished mask_normalize_inputs.py"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "4. PROCESS MODEL INPUTS" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo "Starting make_lr_inputs.py..."
if ! python -m 4-process-inputs/make_lr_inputs.py; then 
    echo "ERROR: Failed to run 4-process-inputs/make_lr_inputs.py"
    exit 1
fi

echo "Finished make_lr_inputs.py," 
echo "Starting make_cnn_inputs.py..."
if ! python -m 4-process-inputs/make_cnn_inputs.py; then 
    echo "ERROR: Failed to run 4-process-inputs/make_cnn_inputs.py"
    exit 1
fi

echo "Finished make_cnn_inputs.py" 
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "MODEL TRAINING"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "5. PERSISTENCE" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
MODEL_STR="ps"

# Define model script directory string
DIR_STR="5-ps"

echo "Starting $MODEL_STR.py..."
if ! python -m $DIR_STR/$MODEL_STR.py; then 
    echo "ERROR: Failed to run $DIR_STR/$MODEL_STR.py"
    exit 1
fi

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate/quick_eval.py; then 
    echo "ERROR: 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR train and quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "6. LINEAR REGRESSION" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
MODEL_STR="lr_cf"

# Define model script directory string
DIR_STR="6-lr"

echo "Starting $MODEL_STR.py..."
if ! python -m $DIR_STR/$MODEL_STR.py; then 
    echo "ERROR: Failed to run $DIR_STR/$MODEL_STR.py"
    exit 1
fi

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate/quick_eval.py; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR train and quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "7. WEIGHTED LINEAR REGRESSION" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
MODEL_STR="lr_wtd_cf"

# Define model script directory string
DIR_STR="7-lr-weighted"

echo "Starting $MODEL_STR.py..."
if ! python -m $DIR_STR/$MODEL_STR.py; then 
    echo "ERROR: Failed to run $DIR_STR/$MODEL_STR.py"
    exit 1
fi

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate/quick_eval.py; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR train and quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SWITCH CONDA ENVIRONMENT TO PYTORCH
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Deactivate environment
conda deactivate

# Activate Pytorch Environment
conda activate torch_env

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "8. CNN" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
MODEL_STR="cnn_pt"

# Define model script directory string
DIR_STR="8-cnn"

echo "Starting $MODEL_STR.py..."
if ! python -m $DIR_STR/$MODEL_STR.py; then 
    echo "ERROR: Failed to run $DIR_STR/$MODEL_STR.py"
    exit 1
fi

echo "FINISHED $MODEL_STR train"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "9. WEIGHTED CNN" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
MODEL_STR="cnn_wtd_pt"

# Define model script directory string
DIR_STR="9-cnn-weighted"

echo "Starting $MODEL_STR.py..."
if ! python -m $DIR_STR/$MODEL_STR.py; then 
    echo "ERROR: Failed to run $DIR_STR/$MODEL_STR.py"
    exit 1
fi

echo "FINISHED $MODEL_STR train"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SWITCH CONDA ENVIRONMENT BACK FOR PLOTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Deactivate environment
conda deactivate

# Activate Pytorch Environment
conda activate seaice

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# QUICK EVAL PLOTS FOR CNN MODELS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
MODEL_STR="cnn_pt"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate/quick_eval.py; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

# Define model type string for script and quick evaluation plots
MODEL_STR="cnn_wtd_pt"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate/quick_eval.py; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "FINISHED ALL ML MODEL TRAINING AND QUICK EVAL"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

exit 0