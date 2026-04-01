import io
import numpy as np
from pathlib import Path
import time
import xarray as xr
from typing import Tuple, Generator, TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig
    from requests import Session

from earthdata_auth import create_earthdata_session
from urls import IceVelURLBuilder

def main(cfg: PipelineConfig):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')
    
    # Define raw data destination file name
    filename = 'ice_vel_raw_ppv4_ease.npz'
    
    # Create Nasa Earth Data session
    earth_data_session = create_earthdata_session()

    # Initialize url builder
    url_builder = IceVelURLBuilder(cfg)

    # Iterate thorugh URLs from generator
    for url in url_builder.build():

        # Load current url data
        ui, vi, ri, time = load_icevel_data(url, earth_data_session)



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
        
        except Exception:
            print(f'Attempt {attempt +1} failed: {Exception}')

            # Wait for delay and retry if not all attempts used
            if attempt < retries - 1:
                time.sleep(delay)
            
            # Raise exeption
            else:
                raise


def load_icevel_data(url: str, session: Session) -> Tuple[np.NDarray, ...]:
    """
    
    """

    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:

        ui = ds['u'].values
        vi = ds['v'].values
        ri = ds['icemotion_error_estimates'].values
        time = ds['time'].values

    return ui, vi, ri, time


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)