import os
from .param import (
    HEM,
    START_YEAR, END_YEAR,
    TIMESTAMP_IN, TIMESTAMP_OUT, TIMESTAMP_COORD,
)

# Get current script directory
script_dir = os.path.dirname(__file__)

# Get root directory (one level above)
root = os.path.abspath(
    os.path.join(
        script_dir,
        '..',
    )
)

# Define coordinate variable source path
PATH_SOURCE_COORD = os.path.abspath(
    os.path.join(
        root,
        'data',
        'coordinates',
        HEM,
        TIMESTAMP_COORD,
    )
)

# Define masked-normalized data source path
PATH_SOURCE = os.path.abspath(
    os.path.join(
        root,
        'data',
        'mask-norm',
        HEM,
        TIMESTAMP_IN,
    )
)

# Define lr input destination path
PATH_DEST_LR = os.path.abspath(
    os.path.join(
        root,
        'data',
        'lr-input',
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST_LR, exist_ok = True)

# Define cnn destination path
PATH_DEST_CNN = os.path.abspath(
    os.path.join(
        root,
        'data',
        'cnn-input',
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST_CNN, exist_ok = True)

# Define string for end of coordinate data file
FSTR_END_IN_COORD = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN_COORD}"

# Define string for end of input regrid data file
FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}"

# Define string for end of output data
FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}"