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


# Define raw data source path
PATH_SOURCE = os.path.abspath(
    os.path.join(
        data_root,
        'raw',
        HEM,
        TIMESTAMP_IN,
    )
)

# Define regrid destination path
PATH_DEST = os.path.abspath(
    os.path.join(
        data_root,
        'regrid',
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok = True)

# Define coordinate destination path
PATH_DEST_COORD = os.path.abspath(
    os.path.join(
        data_root,
        'coordinates',
        HEM,
        TIMESTAMP_OUT,
    )
)

# Create the destination directory if it doesn't already exist
os.makedirs(PATH_DEST_COORD, exist_ok = True)

# Define string for end of input data file
FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}"

# Define string for end of output data file
FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}"