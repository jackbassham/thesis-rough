import io
import numpy as np
import numpy.typing as npt
from pathlib import Path
import time
import xarray as xr
from typing import Tuple, Generator, TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig
    from requests import Session

from earthdata_auth import create_earthdata_session
from urls import IceVelURLBuilder

# TODO before memmaps, log and save array shapes in metadata
# TODO chunked processing for memory
# OR TODO np memmaps (through entire ml pipeline) (need metadata)
# TODO crop with buffer before downloading to save disk space
# TODO add progress tracking and failure recovery to track each file download success


def main(cfg: PipelineConfig):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')
    
    # Define raw data destination file name
    filename = 'ice_vel_raw_ppv4_ease.npz'
    
    # Create Nasa Earth Data session
    earth_data_session = create_earthdata_session()

    # Initialize url builder
    url_builder = IceVelURLBuilder(cfg)

    # Get iterable of URLs from builder
    url_iter = url_builder.build()

    # Get first URL from builder
    first_url = next(url_iter)

    # Get lat and lon data from the first url in the iterator
    lat, lon = load_lat_lon(first_url, earth_data_session)

    # Initialize lists for dataset variables
    ui_all, vi_all, ri_all, time_all = [], [], [], []

    # Iterate thorugh URLs from generator
    for url in [first_url, *url_iter]:

        # Load current url data
        ui, vi, ri, time = load_icevel_data(url, earth_data_session)

        # Append to lists
        ui_all.append(ui)
        vi_all.append(vi)
        ri_all.append(ri)
        time_all.append(time)

    # Concatenate data lists along time dimension
    ui_all = np.concatenate(ui_all, axis = 0)
    vi_all = np.concatenate(vi_all, axis = 0)
    ri_all = np.concatenate(ri_all, axis = 0)
    time_all = np.concatentate(time_all, axis = 0)

    ...


def open_netcdf_from_response(
        url: str, session: Session , retries=3, delay=5
        ) -> xr.Dataset:
    """
    
    """
    
    # Attempt to access file for number of retries
    for attempt in range(retries):

        try:
            # Return xarray dataset object from session response
            return xr.open_dataset(
                url,
                engine='netcdf4',
                backend_kwargs={'session': session}
            )
        
        except Exception as e:
            print(f'Attempt {attempt +1} failed: {e}')

            # Wait for delay and retry if not all attempts used
            if attempt < retries - 1:
                time.sleep(delay)
            
            # Raise exeption
            else:
                raise


def load_icevel_data(url: str, session: Session) -> Tuple[npt.NDarray, ...]:
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


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)