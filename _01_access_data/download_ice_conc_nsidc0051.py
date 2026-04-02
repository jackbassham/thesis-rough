import numpy as np
import numpy.typing as npt
from pathlib import Path
from requests import Session
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from requests import Session

from .earthdata_auth import create_earthdata_session
from .urls import (
    IceConcURLBuilderNSIDC0051,
    PSGridURLBuilderNSIDC0771,
)
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
    filename = 'ice_conc_raw_nsidc0051v2_ps.npz'
    
    # Create Nasa Earth Data session
    earth_data_session = create_earthdata_session()

    # Initialize url builder
    url_builder = IceConcURLBuilderNSIDC0051(cfg)

    # Initialize lists for dataset variables
    ci_all, time_all = [], []

    # Iterate thorugh URLs from generator
    for i, url in enumerate(url_builder.build()):

        # Load current url data
        ci, time = load_iceconc_data(url, earth_data_session)

        # Append to lists
        ci_all.append(ci)
        time_all.append(time)

        # Get lat lon variables and coordinates once from first url
        if i == 0:
            y, x = load_spatial_coordinates(url, earth_data_session)

        # Print step
        print(f'url index {i} loaded')

    # Initialize Polar Stereographic grid URL builder
    url_builder = PSGridURLBuilderNSIDC0771(cfg)

    # Load lat and lon variables once from the polar stereographic grid
    for url in url_builder.build():
        lat, lon = load_lat_lon(url, earth_data_session)    

    # Concatenate data lists along time dimension
    ci_all = np.concatenate(ci_all, axis = 0)
    time_all = np.concatenate(time_all, axis = 0)

    # Convert time to datetime64 object
    time_all = np.array([np.datetime64(t) for t in time_all])

    # Save the data
    np.savez(
        path_raw / filename,
        ci = ci_all,
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

        print(ds_var_names)

        # Match dataset variable names to list of possible names
        names_match = ds_var_names.intersection(possible_names)

        print(names_match)

        if names_match:
            # Extract data with variable name from set
            name = next(iter(names_match))
            ci = ds[name].values
            time = ds['time'].values
            return ci, time
        
        # Handle case where datset name not in set
        else:
            raise ValueError('Sensor specific vairable name not in dataset, inspect dataset')


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)