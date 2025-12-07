#!/bin/zsh


echo "Starting ProcessInputs.py..."
if ! python ProcessInputs.py; then 
    echo "ERROR: Failed to run ProcessInputs.py"
    exit 1
fi

echo echo "Finished ProcessInputs.py, Starting WcnnTorch.py..."
if ! python ./weighted/WcnnTorch.py; then
    echo "ERROR: Failed to run WcnnTorch.py"
    exit 1

echo "All scripts completed."
