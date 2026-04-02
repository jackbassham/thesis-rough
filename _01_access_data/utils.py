import io
import time
from requests import Session
import xarray as xr



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