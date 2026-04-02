import io
import numpy as np
import numpy.typing as npt
from pathlib import Path
from requests import Session
import time
import xarray as xr
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from requests import Session

from .earthdata_auth import create_earthdata_session
from .urls import IceVelURLBuilder

# TODO before memmaps, log and save array shapes in metadata
# TODO chunked processing for memory
# OR TODO np memmaps (through entire ml pipeline) (need metadata)
# TODO crop with buffer before downloading to save disk space
# TODO add progress tracking and failure recovery to track each file download success
# TODO abstract saving function using dict with variable names to variables


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


def open_netcdf_from_response(
        url: str, session: Session, retries=3, delay=5
        ) -> xr.Dataset:
    """
    
    """
    
    # Attempt to access file for number of retries
    for attempt in range(retries):

        try:

            # Get response from session
            response = session.get(url)
            print(f'Attempt {attempt +1} Response {response}')

            # Raise HTTP error if unsucessful
            response.raise_for_status()

            # Return xarray dataset from session response object
            return xr.open_dataset(io.BytesIO(response.content))
        
        except Exception as e:
            print(f'Attempt {attempt +1} failed: {e}')

            # Wait for delay and retry if not all attempts used
            if attempt < retries - 1:
                time.sleep(delay)
            
            # Raise exeption if all attempts used
            else:
                raise


def load_icevel_data(url: str, session: Session) -> Tuple[npt.NDArray, ...]:
    """
    
    """

    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:

        ui = ds['u'].values
        vi = ds['v'].values
        ri = ds['icemotion_error_estimate'].values
        time = ds['time'].values

    return ui, vi, ri, time


def load_lat_lon(url: str, session: Session):
    """
    
    """

    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values

    return lat, lon


def load_spatial_coordinates(url: str, session: Session):
    """
    
    """
    
    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:
        y = ds["y"].values
        x = ds["x"].values

    return y, x


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)