import numpy as np
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master '<  >.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

TIMESTAMP_IN = os.getenv("TIMESTAMP_IN") # timestamp version of input data

TIMESTAMP_COORD = os.getenv("TIMESTAMP_COORD") # timestamp version of coordinate data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define masked & normalized input data path; relative to current
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        'regrid', 
        HEM,
        TIMESTAMP_IN)
)

# Define path to coordinate variables
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir,
        '..',
        'data',
        'coordinates',
        HEM,
        TIMESTAMP_COORD)
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additonal global variables here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}"
FSTR_END_COORD = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_COORD}"

def main():

    # Load in coordinate variables from motion dataset
    data = np.load(os.path.join(PATH_SOURCE, f'motion_ppv4_{FSTR_END_IN}'))
    time = data['time']
    lat = data['lat']
    lon = data['lon']

    # Save coordinate variables in new file
    np.savez(
        os.path.join(PATH_DEST, f'coord_{FSTR_END_COORD}'),
        time = time,
        lat = lat,
        lon = lon
    )

    return

if __name__ == "__main__":
    main()
