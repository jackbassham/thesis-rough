from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Tuple


# NOTE: Tuples changed to dataclasses to avoid confusion when unpacking
# NOTE: Functionality remains in functions to avoid overcomplicating with class methods
# and instantiation


@dataclass
class OldGridProj:
    lat_mesh: npt.NDArray[np.float64]
    lon_mesh: npt.NDArray[np.float64]
    coordinates_are_vectors: bool = False

    def __post_init__(self):

        # If it's specified that the old lat/lon are coordinate vectors 
        # ie: lat[y] and lon[x]
        if self.coordinates_are_vectors:
            # Check that they are coordinate vectors
            self._validate_coordinate_vectors()

            # Convert them to coordinate grids
            # ie: lat[y, x] and lon[y, x]
            self._convert_to_grid()

        else:
            # Check that the old lat/lon coordinate grids are valid
            self._validate_coordinate_grids()


    def _convert_to_grid(self):
        # Create mesh grids from old lon (x) and lat (y)
        lon_mesh, lat_mesh = np.meshgrid(self.lon_mesh, self.lat_mesh)

        # Replace old lat and lon with mesh grids
        self.lat_mesh = lat_mesh
        self.lon_mesh = lon_mesh

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter validation methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _validate_coordinate_vectors(self):
        # Handle case where boolean not set to specify that old lat and lon are 1D coordinate variables
        if self.lat_mesh.ndim != 1 or self.lon_mesh != 1:
            raise ValueError('coordinates_are_vectors=True but old lat/lon are not 1D vectors')
            
    def _validate_coordinate_grids(self):
        
        # Handle case where coordinates are vectors but object specifies grid
        if self.lat_mesh.ndim == 1 or self.lon_mesh.ndim == 1:
            raise ValueError('coordinates_are_vectors=False but old lat/lon are 1D vectors')

        # Handle any other invalid dimension for the lat lon projection variables
        if self.lat_mesh.ndim != 2 or self.lon_mesh.ndim != 2:
            raise ValueError('old lat/lon dimensions are not valid (not 1D vectors or 2D grids), inspect dataset')
        
        if self.lat_mesh.shape != self.lon_mesh.shape:
            raise ValueError('old lat[y,x] and lon[y,x] grids are not the same shape')


@dataclass
class GridSpecs:
    """
    Specifications for new regular lat/lon grid.
    """
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    resolution_km: float

    # NOTE error handling already covered in DataConvig?

    def resolution_degrees_lat(self):

        # Convert latitude resolution to degrees from kilometers
        return self.resolution_km / 111

    def resolution_degrees_lon(self):

        # Get the average latitude of the region
        avg_lat = np.mean(self.lat_bounds)

        # Convert longitude resolution to degrees from kilometers
        # based on average latitude
        return self.resolution_km / (111 * np.cos(np.radians(avg_lat)))


@dataclass
class NewRegGrid:
    lat: npt.NDArray[np.float64]
    lon: npt.NDArray[np.float64]


@dataclass
class InterpIndices:
    jj: int
    ii: int


# FIXME awkward implementation of data class for data used
# throughout pipeline in regrid methods
# add variable_config in _00_config?
@dataclass
class VectorField:
    u: npt.NDArray
    v: npt.NDArray


def main():

    ...


# TODO split up new grid construction and nearest neighbor interpolation
def construct_regular_grid(grid_specs: GridSpecs) -> NewRegGrid:
    """
    
    """

    lat_reg = np.arange(
        grid_specs.lat_bounds[0],
        grid_specs.lat_bounds[1] + grid_specs.resolution_degrees_lat(),
        grid_specs.resolution_degrees_lat()
    )

    lon_reg = np.arange(
        grid_specs.lon_bounds[0],
        grid_specs.lon_bounds[1] + grid_specs.resolution_degrees_lon(),
        grid_specs.resolution_degrees_lon()
    )

    return NewRegGrid(
        lat = lat_reg, 
        lon = lon_reg,
        )


def compute_nearest_neighbor_indices(new_reg_grid: NewRegGrid, old_grid_proj: OldGridProj) -> InterpIndices:
    """
    Returns indices of closest old point projection grid point data[y,x] at lat[y,x]/lon[y,x]
    to each new regular grid point data[lat[y],lon[x]]

    NOTE:
    At each (j, i) iteration of the new grid, vertical and horizontal distances are computed between that new
    gridpoint's latitude and longitude (in degrees) and all old grid latitude and longitudes (in degrees). 
    The absolute distance is taken by adding dx and dy (pythagorean theorem here). 
    Neighbor indices are taken from the minimum absolute distances, and then the minimum of those indices 
    (lower left hand corner) are taken for consistencey.

    At each iteration, these indices are stored in lookup tables (jj, ii) representing the vertical and horizontal indices
    of each gridpoint.

    ie:
    jj = [[100, 99, 99, 100],
    [101, 100, 99, 100]]

    ii = [[91, 90, 90, 91].
    [93, 92, 91, 92]]

    So (0,0) on the new grid corresponds to index (100, 91) on the old grid.

    NOTE:
    (j,i) is used against the convention (i,j) to represent (vertical, horizontal) indexing 
    when plotting arrays in Python.
    """

    # Get number of new regular grid lat/lon points
    num_lat = len(new_reg_grid.lat)
    num_lon = len(new_reg_grid.lon)

    # Initialize lookup tables to zeros with native integer type
    jj = np.zeros((num_lat, num_lon), type = np.intp)
    ii = np.zeros((num_lat, num_lon), type = np.intp)

    # Iterate through new regular grid lat/lon points
    for j in range(num_lat):
        for i in range(num_lon):

            # Calculate vertical distance between new lat at current index and all old lats
            dy = (new_reg_grid.lat[j] - old_grid_proj.lat_mesh)**2

            # Calculate horizontal distance between new lon at current index and all old lons
            # NOTE Considering multiple cases due to periodicity of longitude
            # although -180 to 180 longitude should be enforced in config, 
            # periodicity used here for function reusability
            dx1 = (new_reg_grid.lon[i] - old_grid_proj.lon_mesh)**2
            dx2 = (new_reg_grid.lon[i] - old_grid_proj.lon_mesh + 360)**2
            dx3 = (new_reg_grid.lon[i] - old_grid_proj.lon_mesh - 360)**2

            # Take the minimum horizontal distance of the cases
            dx = np.minimum(dx1, np.minimum(dx2, dx3))

            # Find the absolute distances between points (by pythagorean theorem)
            ds = dx + dy

            # Find indices of the minimum absolute distances
            i_neighbors = np.where(ds == np.min(ds))

            # Take minimum of the minimums (ie: lower left corner) for consistency
            jj[j,i] = np.min(i_neighbors[0])
            ii[j,i] = np.min(i_neighbors[1])

            print(type(jj))
            print(type(ii))
            print(np.shape(ii))
            print(np.shape(jj))
            print(ii[j,i])
            print(jj[j,i])

    return InterpIndices(
        jj = jj,
        ii = ii,
        )


def rotate_to_East_North(
    u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], 
    old_grid_proj: OldGridProj, hemisphere: str
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Rotates vector components to positive East/ North orientation.
    https://nsidc.org/data/user-resources/help-center/how-convert-horizontal-and-vertical-components-east-and-north
    """

    # Get rotation angle from longitude
    theta = np.radians(old_grid_proj.lon_mesh)

    # Rotate u and v vector components for Southern Hemisphere
    if hemisphere.lower().strip() == 'south':
        u_rot = u * np.cos(theta) - v * np.sin(theta) 
        v_rot = u * np.sin(theta) + v * np.cos(theta)

    # Rotate u and v vector components for Northern Hemisphere
    elif hemisphere.lower().strip() == 'north':
        u_rot = u * np.cos(theta) + v * np.sin(theta)
        v_rot = -u * np.sin(theta) + v * np.cos(theta)

    else:
        # Handle case where hemisphere is not entered corectly
        raise ValueError('"hemisphere" string must be "south" or "north"')

    # FIXME dataclass to return vectorfield instead of Tuple to avoid mixup in returns?
    return u_rot, v_rot


def regrid_data(data: npt.NDArray[np.float64], interp_indices: InterpIndices):
    """
    Regrids data using vectorized indexing into old data with indices jj and ii 
    that have same shape as new regular grid.
    """

    # FIXME use list comprehension for all *data with 'yield' instead of 'return' for gracefulness?

    # Handle case where data is not shaped [time, lat, lon]
    if data.ndim != 3:
        raise ValueError('data must be shaped [time, lat, lon]')

    return data[:,interp_indices.jj, interp_indices.ii]


if __name__ == '__main__':
    main()