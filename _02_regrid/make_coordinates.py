import numpy as np
from typing import TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from helpers import load_npz_data

def main(cfg: PipelineConfig):

    # Load regrid data (source and destination) path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Define list of data file names
    filenames = Path(
        [
        'ice_vel_regrid_ppv4.npz',
        'ice_conc_regrid_nimbus7.npz',
        'wind_regrid_jra55.npz',
        ]
    )

    # Define list of dataset names
    datanames = [
        'ice_vel',
        'ice_conc',
        'wind',
    ]






    # Load in coordinate variables from motion dataset
    data = np.load(os.path.join(PATH_REGRID, f'motion_ppv4_latlon.npz'))
    time_icevel = data['time']
    lat_icevel = data['lat']
    lon_icevel = data['lon']

    # Load in coordinate variables from wind dataset
    data = np.load(os.path.join(PATH_REGRID, f'wind_jra55_latlon.npz'))
    time_wind = data['time']
    lat_wind = data['lat']
    lon_wind = data['lon']

    # Load in coordinate variables from concentration dataset
    data = np.load(os.path.join(PATH_REGRID, f'con_nimbus7_latlon.npz'))
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

    # Update time to reflect data shift for 'present day' inputs
    time_t0 = time_icevel[1:]

    # Create the destination directory if it doesn't already exist
    os.makedirs(PATH_COORDINATES, exist_ok = True)

    # Save coordinate variables from one dataset in new file
    np.savez(
        os.path.join(PATH_COORDINATES, f'coordinates.npz'),
        time_total = time_icevel,
        time_t0 = time_t0,
        lat = lat_icevel,
        lon = lon_icevel,
    )

    return

def build_coordinate_dict(path_regrid: path, filenames: list, datanames:list) -> dict:
    """
    
    """

    coordinates = [
        load_npz_file
    ]

    return coordinates


def check_coordinates_equal(coord_dict):
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
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)