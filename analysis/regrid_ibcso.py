import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import xarray as xr

from _02_regrid.core_regrid import (
    OldGridProj,
    GridSpecs,
)
from _02_regrid.pipeline_regrid import regrid_dataset

from _00_config.load_config import load_config


def main(cfg):

    print('~~~~~~~~~~~~~~~~REGRID IBCSO~~~~~~~~~~~~~~~~')

    # Define path to ibcso data
    path = 'analysis/IBCSO_v2_bed.nc'

    # Load in dataset in chunks
    ds = load_ibcso_data(path)

    print('~~~~~~~~~~~~~~~~DATA LOADED~~~~~~~~~~~~~~~~')

    lat, lon = generate_lat_lon(ds)

    print('~~~~~~~~~~~~~~~~LAT LON GENERATED~~~~~~~~~~~~~~~~')

    scalar_field = {'z': ds['z'].values}

    # Instantiate old grid projection object
    old_grid_proj = OldGridProj(
        lat_mesh = lat,
        lon_mesh = lon,
    )

    # Instantiate new grid specifications object
    grid_specs = GridSpecs(
        lat_bounds = cfg.data_config.latitude_bounds,
        lon_bounds = cfg.data_config.longitude_bounds,
        resolution_km = cfg.data_config.grid_resolution,
    )

    # Regrid vector and scalar data and create new grid lat/lon
    _, scalars_regrid, new_reg_grid = regrid_dataset(
        old_grid_proj, grid_specs, 
        cfg.data_config.hemisphere,
        scalar_fields = scalar_field, 
    )

    print('~~~~~~~~~~~~~~~~DATA REGRID~~~~~~~~~~~~~~~~')

    # Plot the regrid data
    plt.pcolormesh(scalars_regrid['z'])
    plt.colorbar()
    plt.savefig('regrid_ibcso.png')

    print('~~~~~~~~~~~~~~~~FINISHED~~~~~~~~~~~~~~~~')


def load_ibcso_data(path):
    ds = xr.open_dataset(path, chunks={'x': 1000, 'y': 1000})
    return ds


def generate_lat_lon(ds):

    transformer = Transformer.from_crs('EPSG:9354', 'EPSG:4326', always_xy=True)

    def transform_func(x, y):
        return transformer.transform(x, y)

    lon, lat = xr.apply_ufunc(
        transform_func,
        ds['x'],
        ds['y'],
        input_core_dims=[['x'], ['y']],
        output_core_dims=[['x'], ['y']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )

    return lat, lon


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)