import numpy as np
from pathlib import Path
import xarray as xr
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

def main(cfg: PipelineConfig):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')
    
    # Define raw data destination file name
    filename = 'ice_vel_raw_ppv4_ease.npz'

    ...


def construct_ice_vel_urls(hemisphere: str, year_range: Tuple[int, int]) -> str:
    """
    Builds url for https file system download of daily ice velocity data:
    'Polar Pathfinder Daily 25 km EASE-Grid Sea Ice Motion Vectors, Version 4'
    DATA SET ID: NSIDC-0116
    DOI: 10.5067/INAWUWO7QH7B
    Access/ Documentation: https://nsidc.org/data/nsidc-0116/versions/4

    NOTE Subject to change, Nasa Earth Data moving to cloud.
    """

    # Define parent url
    parent = 'https://daacdata.apps.nsidc.org/pub/DATASETS/'

    # Map hemisphere to string format in host directory
    hem_dir= {
        'south': 'south',
        'north': 'north',
    }

    # Define dataset directory url
    data_directory = f'nsidc0116_icemotion_vectors_v4/{hem_dir}/daily/'

    # Map hemisphere to string format in file name
    hem_fnam = {
        'south': 'sh',
        'north':  'nh',
    }

    download_urls = []

    # Unpack year range tuple
    start_year, end_year = year_range

    # Iterate through years to create filenames
    for year in range(start_year, end_year):
    
        # Construct filename based on host structure
        filename = f'icemotion_daily_{hem_fnam[hemisphere]}_25km_{year}0101_{year}1231_v04.nc'

        # Construct url
        download_url = parent + data_directory + filename

        # Append to list of urls
        download_urls.append(download_url)

    return download_urls


    
if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)