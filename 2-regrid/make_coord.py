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
    data = np.load(os.path.join(PATH_SOURCE, f'motion_ppv4_latlon_{FSTR_END_IN}.npz'))
    time_icevel = data['time']
    lat_icevel = data['lat']
    lon_icevel = data['lon']

    # Load in coordinate variables from wind dataset
    data = np.load(os.path.join(PATH_SOURCE, f'wind_jra55_latlon_{FSTR_END_IN}.npz'))
    time_wind = data['time']
    lat_wind = data['lat']
    lon_wind = data['lon']

    # Load in coordinate variables from concentration dataset
    data = np.load(os.path.join(PATH_SOURCE, f'con_nimbus7_latlon_{FSTR_END_IN}.npz'))
    time_con = data['time']
    lat_con = data['lat']
    lon_con = data['lon']

    # Assert latitude coordinate variables are consistent accross datasets
    assert_equal(
        {
        "lat_icevel": lat_icevel,
        "lat_wind": lat_wind,
        "lat_con": lat_con
        }
    )

    # Assert longitude coordinate variables are consistent accross datasets
    assert_equal(
        {
        "lon_icevel": lon_icevel,
        "lon_wind": lon_wind,
        "lon_con": lon_con
        }
    )

    # Assert time coordinate variables are consistent accross datasets
    assert_equal(
        {
        "time_icevel": time_icevel,
        "time_wind": time_wind,
        "time_con": time_con
        }
    )

    # Save coordinate variables from one dataset in new file
    np.savez(
        os.path.join(PATH_DEST, f'coord_{FSTR_END_COORD}'),
        time = time_icevel,
        lat = lat_icevel,
        lon = lon_icevel
    )

    return

def assert_equal(coord_dict):
    """
    Asserts arrays are equal
    
    Takes input 'coord_dict' mapping name to array
        Example: {"a": a, "b": b, "c": c}

    If arrays are equal, program continues.
    If not, program strops and value error is raised with print statement.
    """

    # Pull variable names from dict keys
    names = list(coord_dict.keys())
    for i in range (len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i] ,names[j]
            if not np.array_equal(coord_dict[a], coord_dict[b]):
                raise ValueError(f"ERROR: {a} and {b} not equal")
            

if __name__ == "__main__":
    main()
