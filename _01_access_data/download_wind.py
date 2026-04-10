import cdsapi
import matplotlib.pyplot as plt
import xarray as xr

# Example code at:
# https://cds.climate.copernicus.eu/how-to-api


def main(cfg):

    # Define dataset
    dataset = "reanalysis-era5-single-levels"

    client = cdsapi.Client()


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




def get_cds_request(year: str, 
                latitude_bounds: str,
                longitude_bounds: str):
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
    "year": [year]
    ,
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
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",

    # Area with buffer for regrid
    "area": [
        str(latitude_bounds[1] + 5),
        str(longitude_bounds[0] + 5),
        str(latitude_bounds[0] + 5),
        str(longitude_bounds[1] + 5) 
    ]
    }


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)