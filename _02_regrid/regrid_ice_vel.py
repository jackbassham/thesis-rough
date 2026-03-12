import numpy as np
from pathlib import Path
import sys
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

from core_regrid import(
    OldGridProj,
    GridSpec,
    construct_regular_grid,
    compute_nearest_neighbor_indices,
    rotate_to_East_North,
    regrid_data,
)

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

    # Instantiate old grid projection object
    old_grid_proj = OldGridProj(
        lat_mesh = lat,
        lon_mesh = lon,
    )

    # Instantiate grid specification object
    grid_spec = GridSpec(
        lat_bounds = cfg.data_config.latitude_bounds,
        lon_bounds = cfg.date_config.lon_bounds,
        resolution_km = cfg.data_config.grid_resolution,
    )

    # Construct new regular lat/lon grid
    new_reg_grid = construct_regular_grid(grid_spec)

    # Compute nearest neighbor interpolation indices using new and old grids
    interp_indices = compute_nearest_neighbor_indices(new_reg_grid, old_grid_proj)

    # Rotate vector components to positive East North from x and y
    ui_rot, vi_rot = rotate_to_East_North(
        ui, vi,
        old_grid_proj,
        cfg.data_config.hemisphere,
    )

    # Regrid data using nearest neighbor interpolation indices
    ui_regrid = regrid_data(ui_rot, interp_indices)
    vi_regrid = regrid_data(vi_rot, interp_indices)
    ri_regrid = regrid_data(ri, interp_indices)

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
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)