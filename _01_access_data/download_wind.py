import cdsapi
import helpers
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
from pathlib import Path
import time
from tqdm import tqdm
from typing import Tuple
import xarray as xr

# Example code at:
# https://cds.climate.copernicus.eu/how-to-api


def main(cfg):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_raw)
    
    # Define raw data destination file name
    filename = cfg.dataset_config.build_filename(
        cfg.dataset_config.wind,
        'raw',
    )

    # Load daily ERA5 data into dict of numpy arrays
    data = load_daily_era5_wind(
        path_raw,
        cfg.data_config.year_range,
        cfg.data_config.latitude_bounds,
        cfg.data_config.longitude_bounds,
    )

    # Save ERA5 wind data to npz file
    helpers.save_arrays(path_raw / filename, data)


def add_buffer(coord: int | float, coord_type: str, deg: int = 5):
    """
    
    """

    # Convert to integer
    coord = int(np.round(coord))

    # Expand with buffer
    new_coord = coord - deg if coord < 0 else coord + deg

    # Clip lat values to (-90, 90) if exceding
    if coord_type == 'lat':
        return(max(-90, min(90, new_coord)))
    # Clip lon values to (-180, 180) if exceding
    elif coord_type == 'lon':
        return(max(-180, min(180, new_coord)))
    # Raise exception if invalid 'coordinate type'
    else:
        raise ValueError('Invalid coord_type, enter argument "lat" or "lon"')


def get_3hrly_cds_request(year: str, 
                latitude_bounds: int | float,
                longitude_bounds: int | float,):
    """
    
    """

    client = cdsapi.Client()

    dataset = 'reanalysis-era5-single-levels'

    request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind"
    ],
    "year": [year],
    
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],

    # 3-hourly data
    "time": [
        "00:00",
        "03:00", 
        "06:00", 
        "09:00", 
        "12:00", 
        "15:00", 
        "18:00", 
        "21:00", 
    ],

    "data_format": "grib",
    "download_format": "unarchived",

    # Define lat/ lon box with 5 degree buffer
    "area": [
        str(add_buffer(latitude_bounds[1], coord_type = 'lat')), # North
        str(add_buffer(longitude_bounds[0], coord_type = 'lon')), # West
        str(add_buffer(latitude_bounds[0], coord_type = 'lat')), # South
        str(add_buffer(longitude_bounds[1], coord_type = 'lon')) # East
    ]
    }

    return request


# TODO abstract functions
def load_daily_era5_wind(
        path: Path,
        year_range: Tuple[int, int], 
        latitude_bounds: Tuple[int | float, int | float], 
        longitude_bounds: Tuple[int | float, int | float],
        retries: int = 3,
        delay: int = 5
        ) -> dict[npt.NDArray[np.floating]]:
    """
    
    """

    # Define dataset
    dataset = "reanalysis-era5-single-levels"

    # Instantiate CDS API client
    client = cdsapi.Client()

    # Define target path for a temporary file
    target = path / 'download.grib'

    # Create array of download years
    years = np.arange(year_range[0], year_range[1]+1)

    # Initialize a dict container for the data
    data = {
        'ua': [],
        'va': [],
        'time': [],
    }

    # Loop through years
    for i, year in enumerate(tqdm(years, desc = 'Years')):

        for attempt in range(retries):
            try:

                # Try to make a request for a year's worth of 3hrly data
                request = get_3hrly_cds_request(
                year, latitude_bounds, longitude_bounds
                )
                # Download the data into temporary grib file
                client.retrieve(dataset, request, target)

            except Exception as e:
                print(f'Attempt {attempt + 1} failed: {e}')

                # Wait for delay and retry if not all attempts used
                if attempt < retries - 1:
                    time.sleep(delay)
                
                # Raise exeption if all attempts used
                else:
                    raise

        # Load the grib dataset file with xarray and resample to daily means
        with xr.open_dataset(target, engine = 'cfgrib').resample(time='1D').mean() as ds_daily:

            # Store lat and lon coordinate variables for the first year
            if i == 0:
                data['lat'] = ds_daily.latitude
                data['lon'] = ds_daily.longitude

            # Append data to variable lists as numpy objects
            data['ua'].append(ds_daily.u10.values)
            data['va'].append(ds_daily.v10.values)
            data['time'].append(ds_daily.time.values)

    # Delete the temporary target file
    os.remove(target)

    # Concatentate the variables along the time dimension
    data['ua'] = np.concatenate(data['ua'], axis = 0)
    data['va'] = np.concatenate(data['va'], axis = 0)
    data['time'] = np.concatenate(data['time'], axis = 0)

    return data


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)