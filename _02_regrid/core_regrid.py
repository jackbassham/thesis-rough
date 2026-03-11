from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


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
            self.convert_to_grid()

        else:
            # Check that the old lat/lon coordinate grids are valid
            self._validate_coordinate_grids()


    def convert_to_grid(self):
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


    # TODO convert old grid coordinate variables to lat lon mesh grid (JRA55)


    # TODO boolean for case when old lat and lon are not mesh grid?

    


    def __post_init__(self):
        
        ...
    

@dataclass
class GridSpec:

    # TODO: specification for new grid (resolution, bounds)
    ...

    # TODO ?? resolution, convert to degrees within spec?
    resolution_km: float


    def _convert_res_to_degrees(self):

        ...



@dataclass
class NewRegGrid:
    lat: npt.NDArray[np.float64]
    lon: npt.NDArray[np.float64]



        


@dataclass
class InterpIndices:

    ...




# FIXME move these to OldProjectedGrid and NewRegularGrid
@dataclass
class NewGridBounds:
    lat: Tuple[float, float]
    lon: Tuple[float, float]


@dataclass
class NewGridResolution:
    lat_deg: float
    lon_deg: float


@dataclass
class NewGridCoordinates:
    lat: npt.NDArray[np.float64]
    lon: npt.NDArray[np.float64]


def main():

    ...


# TODO split up new grid construction and nearest neighbor interpolation

def nearest_neighbor_interpolation(
    grid_resolution_deg: Tuple[float, float], latitude_bounds: Tuple[float, float], longitude_bounds: Tuple[float, float], 
    old_latitude, old_longitude
    ):
    """
    
    """

    # Unpack resolution for latitude and longitude grid
    latitude_resolution_deg, longitude_resolution_deg = grid_resolution_deg

    # Unpack latitude tuple
    min_latitude, max_latitude = latitude_bounds

    # Unpack longitude tuple
    min_longitude, max_longitude = longitude_bounds

    # Initialize new longitude coordinate array
    new_latitude = np.arange(min_latitude, max_latitude + latitude_resolution_deg, latitude_resolution_deg)

    # Initialize new latitude coordinate array
    new_longitude = np.arange(min_longitude, max_longitude + longitude_resolution_deg, longitude_resolution_deg)

    # Get number of new latitude longitude points
    num_latitude = len(new_latitude)
    num_longitude = len(new_longitude)

    # Initialize interpolation indices
    jj = np.empty((num_latitude, num_longitude))
    ii = np.empty((num_latitude, num_longitude))

    # Iterate through each new gridpoint
    for j in range(num_latitude):
        for i in range(num_longitude):

            # Compute meridional distances
            dy = (new_latitude[j] - old_latitude)**2

            # Calculate zonal distances considering periodicity in longitude
            dx1 = (new_longitude[i] - old_longitude) ** 2
            dx2 = (new_longitude[i] - old_longitude + 360) ** 2
            dx3 = (new_longitude[i] - old_longitude - 360) ** 2
            
            # Find the minimum zonal distance
            dx = np.minimum(dx1, np.minimum(dx2, dx3))
        
            # Find minimum shortest distance
            ds = dx + dy

            # Find indices of minimum shortest distance
            i_neighbors = np.where(ds == np.min(ds))

            # Take minimum of the latitude and longitude indices (lower left corner)
            jj[j,i] = np.min(i_neighbors[0])
            ii[j,i] = np.min(i_neighbors[1])

    # Return interpolation indices and new lat and lon coordinate variables
    return jj, ii, new_latitude, new_longitude

    

def convert_km_to_degrees(grid_resolution_km, latitude_bounds, longitude_bounds):
    """
    
    """

    resolution_deg_latitude = grid_resolution_km / 111

    average_latitude = np.nanmean(latitude_bounds)

    resolution_deg_longitude = grid_resolution_km / 111 * np.cos(np.radians(average_latitude))

    return (resolution_deg_latitude, resolution_deg_longitude)