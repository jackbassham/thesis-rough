import os
from .param import (
    HEM,
    START_YEAR, END_YEAR,
    TIMESTAMP_IN, TIMESTAMP_OUT,
)

# # Get current script directory
# script_dir = os.path.dirname(__file__)

# # Get root directory (one level above)
# root = os.path.abspath(
#     os.path.join(
#         script_dir,
#         '..',
#     )
# )

# Define root for data directory
data_root = '/data/globus/jbassham/thesis-rough'

# Define masked-normalized data source path
PATH_SOURCE = os.path.abspath(
    os.path.join(
        data_root,
        'lr-input',
        HEM,
        TIMESTAMP_IN,
    )
)

# Define model output destination path
PATH_DEST = os.path.abspath(
    os.path.join(
        data_root,
        'model-output',
        'lr-cf',
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok = True)

# Define string for end of input regrid data file
FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}"

# Define string for end of output data
FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}"