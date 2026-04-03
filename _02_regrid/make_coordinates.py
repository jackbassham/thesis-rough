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

    # Initialize empty dict for filenames
    filenames = {}

    # Iterate through datastet dicts
    for name, ds in cfg.dataset_config.datasets().items():
        # Build filename for each regrid dataset
        filenames[name] = cfg.dataset_config.build_filename(ds, 'regrid')

    # Load coordinate variables from first dataset as reference
    data_ref = load_npz_data(path_regrid / filenames[0])

    # Build a dictionary of the reference coordinate variables
    coordinate_refs = {
        'lat': data_ref['lat'],
        'lon': data_ref['lon'],
        'time': data_ref['time'],
    }

    # Check that the remaning dataset coordinates match the reference
    check_coordinates_match(path_regrid, coordinate_refs, filenames)

    # Shift time to reflect 'present-day' data shift
    time_t0 = coordinate_refs['time_ref'][1:]

    # Remove last day to reflect 'previous-day' data shift
    time_t1 = coordinate_refs['time_ref'][:-1]

    # Define coordinate variable file name
    filename = 'coordinates.npz'

    # Save the coordinate data
    np.savez(
        path_regrid / filename,
        lat = coordinate_refs['lat'],
        lon = coordinate_refs['lon'],
        time_t0 = time_t0,
        time_t1 = time_t1,
    )


def check_coordinates_match(
        path_source: Path, 
        coordinate_refs: dict[str, npt.NDarray], 
        filenames: list[str],
        ) -> None:
    """

    """

    # Iterate through remaining datasets after reference
    for filename in filenames[1:]:

        # Load in that file's dataset
        data = load_npz_data(path_source / filename)

        # Check that dataset coordinate arrays have same shape and elements as reference
        if not np.array_equal(data['lat'], coordinate_refs['lat_ref']):
            raise ValueError(f'Latitude mistmatch in {filename}')
        
        if not np.array_equal(data['lon'], coordinate_refs['lon_ref']):
            raise ValueError(f'Longitude mismatch in {filename}')

        if not np.array_equal(data['time'], coordinate_refs['time_ref']):
            raise ValueError(f'Time mismatch in {filename}')
            

if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)