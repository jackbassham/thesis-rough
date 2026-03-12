import numpy as np
import numpy.typing as npt
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

    # Load coordinate variables from first dataset as reference
    data_ref = load_npz_data(path_regrid / filenames[0])

    coordinate_refs = {
        'lat_ref': data_ref['lat'],
        'lon_ref': data_ref['lon'],
        'time_ref': data_ref['time'],
    }



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

def check_coordinates_match(
        path_source: Path, coordinate_refs: dict[str, npt.NDarray], filenames: list[str],
        ) -> None:
    """

    """

    # Loop through remaining datasets after reference
    for filename in filenames[1:]:

        # Load in that file's dataset
        data = load_npz_data(path_source / filename)

        # Check that dataset coordinates match reference 
        if not np.array_equal(data['lat'], coordinate_refs['lat_ref']):
            raise ValueError(f'Latitude mistmatch in {filename}')
        
        if not np.array_equal(data['lon'], coorinate_refs['lon_ref']:
            raise ValueError(f'Longitude mismatch in {filename}')

        if not np.array_equal(data['time'], coordinate_refs['time_ref']):
            raise ValueError(f'Time mismatch in {filename}')



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