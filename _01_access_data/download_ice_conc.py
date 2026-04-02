import numpy as np
import numpy.typing as npt
from pathlib import Path
from requests import Session
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from requests import Session

from .earthdata_auth import create_earthdata_session
from .urls import IceVelURLBuilder
from .utils import (
    open_netcdf_from_response,
    load_lat_lon,
    load_spatial_coordinates,
)


def main(cfg):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_raw)
    
    # Define raw data destination file name
    filename = 'ice_vel_raw_ppv4_ease.npz'
    
    # Create Nasa Earth Data session
    earth_data_session = create_earthdata_session()

    # Initialize url builder
    url_builder = IceVelURLBuilder(cfg)

    # Initialize lists for dataset variables
    ui_all, vi_all, ri_all, time_all = [], [], [], []

    # Iterate thorugh URLs from generator
    for i, url in enumerate(url_builder.build()):

        # Load current url data
        ui, vi, ri, time = load_icevel_data(url, earth_data_session)

        # Append to lists
        ui_all.append(ui)
        vi_all.append(vi)
        ri_all.append(ri)
        time_all.append(time)

        # Get lat lon variables and coordinates once from first url
        if i == 0:
            lat, lon = load_lat_lon(url, earth_data_session)
            y, x = load_spatial_coordinates(url, earth_data_session)

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


def load_iceconc_data(url: str, session: Session) -> Tuple[npt.NDArray, ...]:
    """
    
    """

    # Define set of possible data variable names
    # NOTE dependent on sensor used for data: '{sensor}_ICECON'
    possible_names = {
        'N07_ICECON',
        'F08_ICECON', 
        'F11_ICECON', 
        'F13_ICECON', 
        'F17_ICECON',
    }

    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:

        # Get set of variable names in dataset
        ds_var_names = set(ds.data_vars)

        # Match dataset variable names to list of possible names
        names_match = ds_var_names.intersection(possible_names)

        if names_match:
            # Extract data with variable name from set
            name = next(iter(names_match))
            ci = ds[name].values
            time = ds['time'].values
            return ci
        
        # Handle case where datset name not in set
        else:
            raise ValueError('Sensor specific vairable name not in dataset, inspect dataset')

    return ci, time


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)