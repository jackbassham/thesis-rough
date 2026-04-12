import cdsapi
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Example code at:
# https://cds.climate.copernicus.eu/how-to-api


def main(cfg):

    # Define dataset
    dataset = "reanalysis-era5-single-levels"

    client = cdsapi.Client()

    print(str(cfg.data_config.latitude_bounds[0]))
    print(str(cfg.data_config.latitude_bounds[1]))
    print(str(cfg.data_config.longitude_bounds[1]))
    print(str(cfg.data_config.longitude_bounds[0]))


    request = get_cds_request(
        '2019', 
        cfg.data_config.latitude_bounds, 
        cfg.data_config.longitude_bounds
        )

    target = 'download.grib'

    client.retrieve(dataset, request, target)

    # Open with array
    ds = xr.load_dataset('download.grib', engine ='cfgrib')

    print('ds objects: {ds}')

    # TODO Plot both varibales




def retrieve_era5_reanalysis(dataset: str, request) -> None:
    """
    
    """


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


def get_cds_request(year: str, 
                latitude_bounds: int,
                longitude_bounds: int,):
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

    # NOTE area has no buffer for regrid, should be ok reg lat/ lon?
    # Need to consider 0 < abs(lat) + buffer < 90
    # and 0 < abs(lon) + buffer < 180
    "area": [
        str(add_buffer(latitude_bounds[1], coord_type = 'lat')), # North
        str(add_buffer(longitude_bounds[0], coord_type = 'lon')), # West
        str(add_buffer(latitude_bounds[0], coord_type = 'lat')), # South
        str(add_buffer(longitude_bounds[1], coord_type = 'lon')) # East
    ]
    }

    return request


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)