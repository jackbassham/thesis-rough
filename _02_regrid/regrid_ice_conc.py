import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from helpers import load_npz_data
from .core_regrid import OldGridProj, GridSpecs
from .pipeline_regrid import regrid_dataset


def main(cfg: PipelineConfig):

    # FIXME repeated logic accross 3

    # Load raw data source path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Load regrid data destination path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_regrid)

    # Define raw data file name
    filename = 'ice_conc_raw_nsidc0051v2_ps.npz'

    # Load raw data file
    data = load_npz_data(path_raw / filename)

    # Load in time variable for saving in new file
    time = data['time']

    # Pack scalar data into dict
    scalar_field = {'ice_conc': data['ci']}

    # Instantiate old grid projection object
    old_grid_proj = OldGridProj(
        lat_mesh = data['lat'],
        lon_mesh = data['lon'],
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
    vectors_regrid, scalars_regrid, new_reg_grid = regrid_dataset(
        old_grid_proj, grid_specs, 
        cfg.data_config.hemisphere,
        scalar_fields = scalar_field, 
    )

    # Unpack scalar from tuple
    ci_regrid = scalars_regrid['ice_conc']

    # Define regrid data file name
    filename = 'ice_conc_regrid_nsidc0051v2.npz'

    # Save the regrid data
    np.savez(
        path_regrid / filename,
        ci = ci_regrid,
        lat = new_reg_grid.lat,
        lon = new_reg_grid.lon,
        time = time,
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)