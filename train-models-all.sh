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
ok "Global variables:"
echo "  HEM           = $HEM"
echo "  YEARS         = $START_YEAR - $END_YEAR"
echo " "

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