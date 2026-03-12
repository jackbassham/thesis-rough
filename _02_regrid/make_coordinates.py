import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from helpers import load_npz_data

def main(cfg: PipelineConfig):

    # Load regrid data (source and destination) path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Define list of data file names
    filenames = [
        'ice_vel_regrid_ppv4.npz',
        'ice_conc_regrid_nimbus7.npz',
        'wind_regrid_jra55.npz',
        ]

    # Define list of dataset names
    datanames = [
        'ice_vel',
        'ice_conc',
        'wind',
    ]


    # # Update time to reflect data shift for 'present day' inputs
    # time_t0 = time_icevel[1:]

    # # Create the destination directory if it doesn't already exist
    # os.makedirs(PATH_COORDINATES, exist_ok = True)

    # # Save coordinate variables from one dataset in new file
    # np.savez(
    #     os.path.join(PATH_COORDINATES, f'coordinates.npz'),
    #     time_total = time_icevel,
    #     time_t0 = time_t0,
    #     lat = lat_icevel,
    #     lon = lon_icevel,
    # )

    return

def build_coordinate_dicts(path_source: Path, filenames: list[str], datanames: list[str]) -> dict:
    """
    
    """

    for filename in filenames:

        data = load_npz_data(path_source / filename)

        lats = [{'name'}]

    variables = ['lat']

    coordinates = [
        load_npz_data()
    ]

    return lats, lons, times


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