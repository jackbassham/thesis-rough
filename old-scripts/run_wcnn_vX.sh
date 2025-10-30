#!/bin/zsh

VERSION="V6"

echo "Starting ProcessInputs${VERSION}.py..."
python /home/jbassham/jack/script/ProcessInputs${VERSION}.py

echo "Finished ProcessInputs${VERSION}.py Starting CNNProcessInputs${VERSION}.py..."
python /home/jbassham/jack/script/CNNProcessInputs${VERSION}.py

echo "Finished CNNProcessInputs${VERSION}.py Starting CNNWeightdTorch${VERSION}.py..."
python /home/jbassham/jack/script/CNNWeightedTorch${VERSION}.py

echo "All scripts completed."
