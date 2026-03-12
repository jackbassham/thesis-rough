import numpy.typing as npt
from typing import Tuple, Optional, TYPE_CHECKING

from core_regrid import(
    OldGridProj,
    GridSpecs,
    construct_regular_grid,
    compute_nearest_neighbor_indices,
    rotate_to_East_North,
    regrid_data,
)


def regrid_dataset(
        old_grid_proj: OldGridProj,
        grid_specs: GridSpecs,
        hemisphere: str,
        scalar_fields: Optional[dict[str, npt.NDarrsy]] = None,
        vector_fields: Optional[dict[str, Tuple[npt.NDarray, npt.NDarray]]] = None,
        rotate_vectors: bool = False,
):
    
    # Construct new regular lat/lon grid
    new_reg_grid = construct_regular_grid(grid_specs)

    # Compute nearest neighbor interpolation indices using new and old grids
    interp_indices = compute_nearest_neighbor_indices(new_reg_grid, old_grid_proj)

    # Initialize empty dict to store regrid vector field data names and tuples (u, v)
    vectors_regrid = {}

    if vector_fields is not None:

        # Iterate through vector field tuples stored in dict
        for name, (u, v) in vector_fields.items():
            
            # If vector fields call for rotation to postive East/North from x/y
            if rotate_vectors:
                # Rotate vector components
                (u, v) = rotate_to_East_North(
                    u, v, old_grid_proj, hemisphere
            )

            # Regrd each vector component in vector field and store in dict
            vectors_regrid[name] = (
                regrid_data(u, interp_indices),
                regrid_data(v, interp_indices),
            )

    else: 
        # If no vector entries, vectors_regrid remains emtpy
        pass

    # Initialize empty dict to store regrid scalar field data
    scalars_regrid = {}

    if scalar_fields is not None:
        # Regrid scalar data and store names and data in dict
        for name, data in scalar_fields.items():
            scalar_fields[name] = (regrid_data(data, interp_indices))

    else:
        # If no scalar entries, scalars_regrid remains emtpy
        pass

    return vectors_regrid, scalars_regrid, new_reg_grid


