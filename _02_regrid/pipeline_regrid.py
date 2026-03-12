import numpy.typing as npt
from typing import Tuple, TYPE_CHECKING

from core_regrid import(
    OldGridProj,
    GridSpecs,
    construct_regular_grid,
    compute_nearest_neighbor_indices,
    rotate_to_East_North,
    regrid_data,
)


def regrid_dataset(
        scalar_fields: dict[str, npt.NDarrsy],
        vector_fields: dict[str, Tuple[npt.NDarray]],
        old_grid_proj: OldGridProj,
        grid_specs: GridSpecs,
        hemisphere: str,
):
    
    # Construct new regular lat/lon grid
    new_reg_grid = construct_regular_grid(grid_specs)

    # Compute nearest neighbor interpolation indices using new and old grids
    interp_indices = compute_nearest_neighbor_indices(new_reg_grid, old_grid_proj)

    # Initialize empty dict to store rotated vector field data names and tuples (u, v)
    vectors_rotated = {}
    # Iterate through vector field tuples stored in dict
    for name, (u, v) in vector_fields.items():
        # Rotate vector components to positive East North from x and y and store in dict
        vectors_rotated[name] = rotate_to_East_North(
            u, v, old_grid_proj, hemisphere
        )

    # Initialize empty dict to store regrid vector field data names and tuples (u, v)
    vectors_regrid = {}
    # Iterate through vector field tuples stored in dict
    for name, (u, v) in vectors_rotated.items():
        # Regrd each vector component in vector field
        vectors_regrid[name] = (
            regrid_data(u, interp_indices),
            regrid_data(v, interp_indices),
        )

    # Regrid scalar data and store names and data in dict
    scalars_regrid = {
        name: regrid_data(data, interp_indices)
        for name, data in scalar_fields.items()
    }

    return vectors_regrid, scalars_regrid, new_reg_grid


