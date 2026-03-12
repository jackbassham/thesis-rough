import numpy as np
from pathlib import Path
import sys
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from core_regrid import OldGridProj, GridSpecs
from pipeline_regrid import regrid_dataset

def main(cfg: PipelineConfig):

    # Load raw data source path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Load regrid data destination path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_regrid)

    # Define raw data file name
    filename = 'ice_vel_raw_ppv4_ease.npz'

    # Try to load the raw data
    try:
        data = np.load(path_raw / filename)

    # Handle case where file does not exist
    except FileNotFoundError:
        sys.exit(f'File at {path_raw / filename} not found')

    # Access variables from data
    ui = data['ui']
    vi = data['vi']
    ri = data['ri']
    lat = data['lat']
    lon = data['lon']
    time = data['time']

    # Pack vector component tuple data into dict
    vectors = {'ice_vel': (ui, vi)}

    # Pack scalar data into dict
    scalars = {'ri': ri}

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

    vectors_regrid, scalars_regrid, new_reg_grid = regrid_dataset(
        vectors, scalars, 
        old_grid_proj, grid_specs, 
        cfg.data_config.hemisphere
    )

    # Unpack vectors from tuple
    ui_regrid, vi_regrid = vectors_regrid['ice_vel']


    # Define regrid data file name
    filename = 'ice_vel_regrid_ppv4.npz'

    # Save the regrid data
    np.savez(
        path_regrid / filename,
        ui = ui_regrid,
        vi = vi_regrid,
        ri = scalars_regrid['ri'],
        lat = new_reg_grid.lat,
        lon = new_reg_grid.lon,
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)