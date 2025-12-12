import os
from .param import (
    HEM,
    START_YEAR, END_YEAR,
    TIMESTAMP_OUT
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

# Define download destination path
PATH_DEST = os.path.abspath(
    os.path.join(
        root,
        'data',
        'raw',
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok = True)

# Define string for end of data file
FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}"