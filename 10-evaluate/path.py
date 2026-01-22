import os

from .param import (
    HEM,
    START_YEAR, 
    END_YEAR,
    MODEL_STR,
    MODEL_DIR,
    TIMESTAMP_MODEL,
    TIMESTAMP_COORD,
    TIMESTAMP_R,
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

# Define model output source path
PATH_SOURCE = os.path.abspath(
    os.path.join(
        root,
        'data',
        'model-output',
        MODEL_DIR,
        HEM,
        TIMESTAMP_MODEL,
    )
)

# Define quick evaluation plots destination path
PATH_DEST = os.path.abspath(
    os.path.join(
        root,
        'plots',
        'quick-eval',
        MODEL_DIR,
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok = True)

PATH_COORD = os.path.abspath(
    os.path.join(
        root,
        'data',
        'coordinates',
        HEM,
        TIMESTAMP_COORD,
    )
)

PATH_R = os.path.abspath(
    os.path.join(
        root,
        'data',
        'lr-input',
        HEM,
        TIMESTAMP_R,
    )
)

# Define string for end of model output file
FSTR_END_MODEL = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_MODEL}"

# Define string for end of plot destination file
FSTR_END_DEST = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}

# Define string for end of coordinate data file
FSTR_END_COORD = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_COORD}"

# Define string for end of uncertaity data file
FSTR_END_R = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_R}"