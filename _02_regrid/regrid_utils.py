import numpy as np
from typing import Tuple

def main():

    ...


def nearest_neighbor_interpolation(
    grid_resolution_deg: Tuple[float, float], latitude_bounds: Tuple[float, float], longitude_bounds:Tuple[float, float], 
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