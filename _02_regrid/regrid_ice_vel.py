import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from helpers import load_npz_data
from core_regrid import OldGridProj, GridSpecs
from pipeline_regrid import regrid_dataset

def main(cfg: PipelineConfig):

    # FIXME repeated logic accross 3

    # Load raw data source path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Load regrid data destination path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_regrid)

    
    # FIXME master dict or dataclass with filenames
    # Define raw data file name
    filename = 'ice_vel_raw_ppv4_ease.npz'

    # Load raw data file
    data = load_npz_data(path_raw / filename)

    # Load in time variable for saving in new file
    time = data['time']

    # Pack vector component tuple data into dict
    vector_field = {'ice_vel': (data['ui'], data['vi'])}

    # Pack scalar data into dict
    scalar_field = {'ri': data['ri']}

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
        vector_fields = vector_field, 
        scalar_fields = scalar_field, 
        rotate_vectors = True,
    )

    # Unpack vectors from tuple
    ui_regrid, vi_regrid = vectors_regrid['ice_vel']

    # Unpack scalar from tuple
    ri_regrid = scalars_regrid['ri']

    # Define regrid data file name
    filename = 'ice_vel_regrid_ppv4.npz'

    # Save the regrid data
    np.savez(
        path_regrid / filename,
        ui = ui_regrid,
        vi = vi_regrid,
        ri = ri_regrid,
        lat = new_reg_grid.lat,
        lon = new_reg_grid.lon,
        time = time,
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)