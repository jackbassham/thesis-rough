import io
import numpy as np
from pathlib import Path
import xarray as xr
from typing import Tuple, Generator, TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

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

    # Yield current URL from builder
    url = url_builder.build()

    # Get the temporary file from url
    temp_file = get_temp_file(url, earth_data_session)

    ...


def get_temp_file(url: str, session) -> io.BytesIO:
    """
    
    """

    try:
        # Submit a response request
        response = session.get(url, stream = True)

        # Print response status (200 for good response!)
        print(response.status_code)

    except:
        # Handle http errors
        response.raise_for_status()

    # Read response content into temporary BytesIOobject
    return io.BytesIO(response.content)



if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)