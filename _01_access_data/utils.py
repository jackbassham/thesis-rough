import io
import time
import numpy as np
import requests
from typing import Optional
import xarray as xr

#TODO use tqdm for progress


def open_netcdf_from_response(
        url: str, 
        retries: int =3, 
        delay: int =5, 
        session: Optional[requests.Session] = None,
) -> xr.Dataset:
    """
    
    """

    # If custom session not provided (default)
    if session is None:
        # Instantiate default session
        session = requests.Session()
        # Enable ~/.netrc use for authorization
        session.trust_env = True
    
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


def load_lat_lon(url: str, session: Session):
    """
    
    """

    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:
        lat = ds["latitude"].values.astype(np.float32)
        lon = ds["longitude"].values.astype(np.float32)

    return lat, lon


def load_spatial_coordinates(url: str, session: Session):
    """
    
    """
    
    # Attempt to open dataset with xarray
    with open_netcdf_from_response(url, session) as ds:
        y = ds["y"].values.astype(np.float32)
        x = ds["x"].values.astype(np.float32)

    return y, x


def download_with_retries():
    """
    
    """

    