#!/bin/zsh
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# README: run-quick-eval.sh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PURPOSE:
#   Script for running quick eval on all model outputs
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
#   chmod +x run-quick-eval.sh
# 
#   # Run script
#   ./run-quick-eval.sh
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

# Define northern or southern hemisphere
export HEM="nh" # "sh" or "nh"

# Define temporal range (see above for replication)
export START_YEAR=1992 # minimum 1989
export END_YEAR=2020

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INITIALIZE LOG FOR SHELL SCRIPT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOGDIR="logs"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/run_${HEM}_$(date +'%Y%m%d_%H%M').log"

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

# Define environment
conda_env="seaice"

# Activate environment
conda activate $conda_env

echo "Activated Conda Environment '$conda_env'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN SCRIPTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get timestamp from python script
export TIMESTAMP=$(python -m helpers.timestamp)

# Print timestamp
echo "Timestamp recorded:"
echo "  Timestamp     = $TIMESTAMP"
echo " "


# Print global variable selections
echo "Global variables:"
echo "  HEM           = $HEM"
echo "  YEARS         = $START_YEAR - $END_YEAR"
echo " "

# For this master shell script, all input/ output timestamp versions will be the same
# For individual shell scripts (model training, etc, enter appropriate timestamp versions for input data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MODIFY TIMESTAMP FOR PARTIAL RUN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Timestamp version of regrid data
export TIMESTAMP_COORD='01072026_1643'

# Timestamp version for model outputs
export TIMESTAMP_MODEL='01212026_1222'

# Timestamp version of uncertainty (from lr input)
export TIMESTAMP_R='01212026_1222'

export TIMESTAMP_OUT=$TIMESTAMP

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "1. PERSISTENCE" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
export MODEL_STR="ps"

# Define model script directory string
DIR_STR="5-ps"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate.quick_eval; then 
    echo "ERROR: 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "2. LINEAR REGRESSION" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
export MODEL_STR="lr_cf"

# Define model script directory string
DIR_STR="6-lr"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate.quick_eval; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "7. WEIGHTED LINEAR REGRESSION" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
export MODEL_STR="lr_wtd_cf"

# Define model script directory string
DIR_STR="7-lr-weighted"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate.quick_eval; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "3. CNN" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
export MODEL_STR="cnn_pt"

# Define model script directory string
DIR_STR="8-cnn"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate.quick_eval; then 
    echo "ERROR: Failed to run 10-evaluate/quick_eval.py"
    exit 1
fi

echo "FINISHED $MODEL_STR quick eval"
echo "PLOTS at: '/plots/quick-eval/$MODEL_STR/$HEM/$TIMESTAMP'"
echo " "

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "4. WEIGHTED CNN" 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define model type string for script and quick evaluation plots
export MODEL_STR="cnn_wtd_pt"

# Define model script directory string
DIR_STR="9-cnn-weighted"

# Generate quick evaluation plots
echo "Starting quick-eval.py"
if ! python -m 10-evaluate.quick_eval; then 
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