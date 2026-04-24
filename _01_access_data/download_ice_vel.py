import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Tuple

from .urls import IceVelURLBuilder
from .utils import (
    open_netcdf_from_response,
    load_lat_lon,
    load_spatial_coordinates,
)

# TODO before memmaps, log and save array shapes in metadata
# TODO chunked processing for memory
# OR TODO np memmaps (through entire ml pipeline) (need metadata)
# TODO crop with buffer before downloading to save disk space
# TODO add progress tracking and failure recovery to track each file download success
# TODO abstract saving function using dict with variable names to variables

# TODO remove hardcoded authentication and replace with .netrc in home

def main(cfg):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_raw)
    
    # Define raw data destination file name
    filename = cfg.dataset_config.build_filename(
        cfg.dataset_config.ice_vel,
        'raw',
    )

    # Initialize url builder
    url_builder = IceVelURLBuilder(cfg)

    # Initialize lists for dataset variables
    ui_all, vi_all, ri_all, time_all = [], [], [], []

    # Iterate thorugh URLs from generator
    for i, url in enumerate(url_builder.build()):

        # Load current url data
        ui, vi, ri, time = load_icevel_data(url)

        # Append to lists
        ui_all.append(ui)
        vi_all.append(vi)
        ri_all.append(ri)
        time_all.append(time)

        # Get lat lon variables and coordinates once from first url
        if i == 0:
            lat, lon = load_lat_lon(url)
            y, x = load_spatial_coordinates(url)

        # Print step
        print(f'url index {i} loaded')

    # Concatenate data lists along time dimension
    ui_all = np.concatenate(ui_all, axis = 0)
    vi_all = np.concatenate(vi_all, axis = 0)
    ri_all = np.concatenate(ri_all, axis = 0)
    time_all = np.concatenate(time_all, axis = 0)

    # Convert time to datetime64 object
    time_all = np.array([np.datetime64(t) for t in time_all])

    # Save the data
    np.savez(
        path_raw / filename,
        ui = ui_all,
        vi = vi_all,
        ri = ri_all,
        lat = lat,
        lon = lon,
        y = y,
        x = x,
        time = time_all,
    )


def load_icevel_data(url: str) -> Tuple[npt.NDArray, ...]:
    """
    
    """

    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url) as ds:

        ui = ds['u'].values.astype(np.float32)
        vi = ds['v'].values.astype(np.float32)
        ri = ds['icemotion_error_estimate'].values.astype(np.float32)
        time = ds['time'].values.astype('datetime64[D]')

    return ui, vi, ri, time


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)