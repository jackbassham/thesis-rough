import cmocean as cmo
import gc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

TIMESTAMP_IN = os.getenv("TIMESTAMP_IN") # timestamp version of input data

TIMESTAMP_MODEL = os.getenv("TIMESTAMP_MODEL") # timestamp version of model run

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define path to model outputs
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir,
        '..',
        'data',
        HEM,
        TIMESTAMP_MODEL
    )
)



def main():

    # Load in true and predicted model outputs
    data = np.load(os.path.join(PATH_SOURCE, )) 

    return