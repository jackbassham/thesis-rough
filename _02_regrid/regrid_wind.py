import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from helpers import load_npz_data
from .core_regrid import OldGridProj, GridSpecs
from .pipeline_regrid import regrid_dataset

def main(cfg):

    # Load raw source path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Load regrid data destination path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_regrid)

    # Define raw data file name
    filename = 'wind_raw_jra55_gaussian.npz'

    # Load raw data file
    data = load_npz_data(path_raw / filename)

    # Load in time variable for saving in new file
    time = data['time']

    # Pack vector component tuple data into dict
    vector_field = {'wind': (data['ua'], data['va'])}

    # Instantiate old grid projection object
    old_grid_proj = OldGridProj(
        lat_mesh = data['lat'],
        lon_mesh = data['lon'],
        coordinates_are_vectors = True,
    )

    # Load in time variable
    time = data['time']

    # Instantiate new grid specifications object
    grid_specs = GridSpecs(
        lat_bounds = cfg.data_config.latitude_bounds,
        lon_bounds = cfg.data_config.longitude_bounds,
        resolution_km = cfg.data_config.grid_resolution,
    )

    # Regrid vector and scalar data and create new grid lat/lon
    vectors_regrid, _, new_reg_grid = regrid_dataset(
        old_grid_proj, grid_specs, 
        cfg.data_config.hemisphere,
        vector_fields = vector_field,
    )

    # Unpack vectors from tuple
    ua_regrid, va_regrid = vectors_regrid['wind']


    # Define regrid data file name
    filename = 'wind_regrid_jra55.npz'

    # Save the regrid data
    np.savez(
        path_regrid / filename,
        ua = ua_regrid,
        va = va_regrid,
        lat = new_reg_grid.lat,
        lon = new_reg_grid.lon,
        time = time,
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)