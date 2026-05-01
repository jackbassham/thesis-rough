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

    # Define path to ibcso data
    path = 'analysis/IBCSO_v2_bed.nc'

    data = load_ibcso_data(path)

    lat, lon = generate_lat_lon(data['y'], data['x'])

    scalar_field = {'z': data['z']}

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

    # Plot the regrid data
    plt.pcolormesh(scalars_regrid['z'])
    plt.colorbar()
    plt.savefig('regrid_ibcso.png')





def load_ibcso_data(path):

    # Load IBCSO data using xarray
    with xr.open_dataset(path) as ds:

        return {
            'y': ds['y'],
            'x': ds['x'],
            'z': ds['z']
        }


def generate_lat_lon(y, x):

    # Instantiate the transofomer for Polar Sterographic (ESPG:9354) to regular lat/lon (EPSG:4326)
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:9354', always_xy=True)

    # Get coordinate grids
    X, Y = np.meshgrid(x, y)

    # Generate lat/lon from the y/x coordinate grids via the transformer
    lat, lon = transformer.transform(X, Y)

    return lat, lon


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)