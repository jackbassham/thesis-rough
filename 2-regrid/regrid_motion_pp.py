import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO fnam move grid to end of name

# Regrids NSIDC Polar Pathfinder Sea Ice Velocities from EASE to regular lat lon coordinate system
# Vector rotation from vertical/horizontal to N/E from:
## https://nsidc.org/data/user-resources/help-center/how-convert-horizontal-and-vertical-components-east-and-north

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

LAT_LIMITS = [float(x) for x in os.getenv("LAT_LIMITS").split(",")] # South to North latitude bounds, degrees
LON_LIMITS = [float(x) for x in os.getenv("LON_LIMITS").split(",")] # West to East longitude bounds, degrees

RESOLUTION = int(os.getenv("RESOLUTION")) # Grid resolution, km

# Nasa Earthdata login credentials for download
USER = os.getenv("USER") # username
PASS = os.getenv("PASS") # password

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define file name to regrid
FNAM = "motion_ppv4_EASE_{HEM}{START_YEAR}{END_YEAR}.npz"

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define absolute raw data directory source path relative to current
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        HEM, 
        'raw')
)

# Define regrid data destination path relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        HEM, 
        'regrid')
)

# Create the direectory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)


def main():

    # Load original grid .npz variables
    filename = FNAM.format(START_YEAR = START_YEAR, END_YEAR = END_YEAR, HEM = HEM)

    # Attempt to load the original .npz file
    try:
        # Load original .npz file
        data = np.load(os.path.join(PATH_SOURCE, filename), allow_pickle=True)

        # Attempt to access variables
        u_old = data['u'] # horizontal ice velocity (cm/s)
        v_old = data['v'] # vertical ice velocity (cm/s)
        error_old = data['error'] # ice motion error estinamtes
        lat_old = data['lat'] # EASE latitude shaped [x,y]
        lon_old = data['lon'] # EASE longitude shaped [x,y]
        time = data['time'] # time series dates dt64

        # Check the type and format of time_total
        print(type(time))  # Should be <class 'numpy.ndarray'>
        print("time_total dtype:", time.dtype)  # Should be datetime64

        print(f"{filename} loaded successfully.")
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in '{PATH_SOURCE}'.")
    except KeyError as e:
        print(f"Error: Missing expected data key: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")


    # Get indices for new lat lon grid
    jj, ii, lat_new, lon_new = nearest_neighbor_interpolation(RESOLUTION, LAT_LIMITS, LON_LIMITS, lat_old, lon_old)

    # Initialize data arrays for current year
    dims = (len(time), len(lat_new), len(lon_new))
    u_new = np.zeros(dims) # zonal ice velocity
    v_new = np.zeros(dims) # meridional ice velocity
    error_new = np.zeros(dims) # icemotion error estimates

    # Iterate through gridpoints
    for i in range(dims[2]):
        for j in range(dims[1]):
            
            # Extract nearest neighbor indices for each new gridpoint
            iii = ii[j,i]
            jjj = jj[j,i]
            
            if HEM == "sh":
                # SOUTHERN HEMISPHERE vector rotation to East/ North
                u_new[:,j, i] = u_old[:,jjj, iii]*np.cos(np.radians(lon_old[jjj, iii])) - v_old[:,jjj, iii]*np.sin(np.radians(lon_old[jjj,iii]))
                v_new[:,j, i] = u_old[:,jjj, iii]*np.sin(np.radians(lon_old[jjj, iii])) + v_old[:,jjj, iii]*np.cos(np.radians(lon_old[jjj,iii]))
                
            elif HEM == "nh":
                # NORTHERN HEMISPHERE vector rotation to East/ North
                u_new[:,j, i] = u_old[:,jjj, iii]*np.cos(np.radians(lon_old[jjj, iii])) + v_old[:,jjj, iii]*np.sin(np.radians(lon_old[jjj,iii]))
                v_new[:,j, i] = -u_old[:,jjj, iii]*np.sin(np.radians(lon_old[jjj, iii])) + v_old[:,jjj, iii]*np.cos(np.radians(lon_old[jjj,iii]))

            else:
                print("Error: Enter HEM nh or sh")
            
            # Extract data from nearest neighbor index
            error_new[:,j,i] = error_old[:,jjj,iii]

    # Create new filename for regrided lat lon data
    fnam = filename.replace("EASE", "latlon")
    path = os.path.join(PATH_DEST, fnam)

    # Save time series data as npz variables
    np.savez_compressed(path, u = u_new, v = v_new, error = error_new, time = time, lat = lat_new, lon = lon_new)
    print(f"Variables Saved at path {path}")

    # NOTE due to vector rotation from horizontal/ vertical to Zonal/ Meridional
    # regrid components alone look incongruous with original data
    # so it is necessary to compare speeds

    # speed_new = np.sqrt(u_new**2 + v_new**2) # lat lon ice motion speed (cm/s)
    # speed_old = np.sqrt(u_old**2 + v_old**2) # EASE ice motion speed (cm/s)

    # #  Compare regrid and old speed
    # fnam = filename.replace(".npz", "_speed_grid_compare.mp4")
    # save_path = os.path.join(PATH_DEST,fnam)
    # compare_grids(speed_new, lat_new, lon_new, speed_old, lat_old, lon_old,
    #         LAT_LIMITS, LON_LIMITS, time = time, main_title = "Ice Speed", save_path = save_path)

    # # Compare regrid and old error
    # fnam = filename.replace(".npz", "_er_grid_compare.mp4")
    # save_path = os.path.join(PATH_DEST,fnam)
    # compare_grids(error_new, lat_new, lon_new, error_old, lat_old, lon_old, 
    #         LAT_LIMITS, LON_LIMITS, time = time, main_title = "Ice Motion Error", save_path = save_path)


    return  


def nearest_neighbor_interpolation(resolution, lat_limits, lon_limits, lat_old, lon_old):
    """
    Returns new regular lat lon coordinate arrays given input resolution in km (based on polar 
    ice veolocity product) and bounds in degrees
    
    Uses nearest neighbor interpolation to return interplation indices for regridding 
    old data to new regular lat lon grid

    Accounts for periodicity in old longitude values

    New lat lon grid represented by (-90S, 90N) and (-180W, 180E)

    """

    # Convert resolution to degrees latitude)
    yres = resolution / 111   # latitude resolution (deg)
    xres = resolution / (111*np.cos(np.radians((lat_limits[0]+lat_limits[1])/2))) # longitude resolution (deg) based on average latitude

    # Create arrays for new lat and lon grid
    lat_new = np.arange(lat_limits[0], lat_limits[1] + yres, yres) # Latitude
    lon_new = np.arange(lon_limits[0], lon_limits[1] + xres, xres) # Longitude
    nlat = len(lat_new)
    nlon = len(lon_new)
    
    # Initialize arrays for interpolation indices
    jj = np.zeros((nlat,nlon), dtype=int)
    ii = np.zeros((nlat,nlon), dtype=int)

    # Iterate through new grid's lat and lon points
    for j in range(nlat):
        for i in range(nlon):

            # Calculate meridional distances
            dy = (lat_new[j]-lat_old)**2

            # Find absolute value distances of i'th lat from entire lat_ease array and store in array  
            dx = (lon_new[i]-lon_old)**2

            # Calculate zonal distances considering periodicity in longitude
            dx1 = (lon_new[i] - lon_old) ** 2
            dx2 = (lon_new[i] - lon_old + 360) ** 2
            dx3 = (lon_new[i] - lon_old - 360) ** 2
            
            # Find the minimum distance for longitude
            dx = np.minimum(dx1, np.minimum(dx2, dx3))
        
            # Find distances
            ds = dx + dy

            # Find indices of minimum ds value
            i_neighbors = np.where(ds == np.min(ds))

            # Take minium of lat and lon indices (lower left corner) for consistency, store in array
            jj[j,i] = np.min(i_neighbors[0])
            ii[j,i] = np.min(i_neighbors[1])

    # Return interpolation indices and new lat and lon coordinate variables
    return jj, ii, lat_new, lon_new

    
def crop_2Dlatlon(data_old, lat_old, lon_old, lat_limits, lon_limits):
    """
    Crops data with 2D lat and lon variables (where lat and lon are used as coordinate variables)
    """
    
    # Check that lon range in (-180, 180)
    if np.any(lon_old > 180):
        lon_old = np.where(lon_old > 180, lon_old - 360, lon_old)  # Convert from 0-360 to -180-180
    elif np.any(lon_old < -180):
        raise ValueError("Longitude values must be in the range (-180, 180).")

    # * Extract j indices along [0]th dimension
    j = np.unique(np.where((lat_old >= lat_limits[0]) & (lat_old <= lat_limits[1]) & (lon_old >= lon_limits[0]) & (lon_old <= lon_limits[1]))[0])
    # * Extract i indices along [1]th dimension
    i = np.unique(np.where((lat_old >= lat_limits[0]) & (lat_old <= lat_limits[1]) & (lon_old >= lon_limits[0]) & (lon_old <= lon_limits[1]))[1])
    
    lat_crop = lat_old[j,:][:,i]
    lon_crop = lon_old[j,:][:,i]
    data_crop = data_old[:,j,:][:,:,i]

    return data_crop, lon_crop, lat_crop


def animated_time_series(data_values, time = None, 
                        main_title = None, titles = None, y_labels = None, x_labels = None, c_labels = None, 
                        vmin = None, vmax = None, cmap = 'viridis', interval = 200, 
                        save_path = None):
    """
    
    Animates multiple subplots of time series data to check regrid, etc
    
    Input Parameters:
    - data: list of data shaped [time, y, x]
    - y_values, x_values: lists of coordinate variables, ie: lat, lon
    - sup_title: main title for plot
    - titles, y_labels, x_labels: as lists for each subplot
    - c_labels: list of colorbar labels
    - vmin, vmax: min and max colorbar values
    - interval: time delay between frames, ms
    - save_path

    """

    # Number of plots based on number of c data inputs
    nplots = len(data_values)

    # Create subplot grid based on number of plots
    fig, axs = plt.subplots(1, nplots, figsize = (6 * nplots, 6))

    # Handle case for one plot
    if nplots == 1:
        # axs not a list
        axs = [axs]

    # Create default case for titles and lables
    if titles is None:
        titles = [f"Data[{i+1}]" for i in range(nplots)]
    
    if x_labels is None:
        x_labels = ["x"] * nplots

    if y_labels is None:
        y_labels = ["y"] * nplots


    # Create time strings for title if time provided
    if time is not None:
        # Convert numpy.datetime64 to string in format "DD MMM YYYY"
        time_strings = np.array([t.astype('datetime64[D]').astype(str) for t in time])
    
    if c_labels is None:
        c_labels = ["Value"] * nplots

    plot_objs = [] # List to hold plot objects

    # Plot initial frame for each subplot
    for i, data in enumerate(data_values):
        plot_obj = axs[i].pcolormesh(data[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(x_labels[i])
        axs[i].set_ylabel(y_labels[i])
        fig.colorbar(plot_obj, ax = axs[i], label = c_labels[i])
        plot_objs.append(plot_obj)

    # Update frame at each time step
    def update(frame):
        for i, data in enumerate(data_values):
            plot_objs[i].set_array(data[frame].ravel())
            # Create main title if provided

        # Create main title based on conditions if provided
        if main_title is not None:
            if time is not None:
                # Create array of date strings if time provided
                fig.suptitle(f"{main_title} {time_strings[frame]}", fontsize = 16, fontweight = 'bold')
            else:
                fig.suptitle(main_title, fontsize = 16, fontweight = 'bold')
        elif time is not None:
            fig.suptitle(time_strings[frame], fontsize = 16, fontweight = 'bold')

        return plot_objs
        
    # Create animation
    # Number of frames based on first data's time dimension
    plot_animated = animation.FuncAnimation(fig, update, frames=data_values[0].shape[0], interval=interval)

    # Save animation if save path exists
    if save_path:
        writer = animation.FFMpegWriter()
        plot_animated.save(save_path, writer = writer)

    return plot_animated


def compare_grids(data_new, lat_new, lon_new, data_old, lat_old, lon_old,  
                  lat_limits, lon_limits, time = None, main_title = None, save_path = None):
    """

    """

    # Crop old data to bounds used for regrid data
    data_old_crop, lat_old_crop, lon_old_crop =  crop_2Dlatlon(data_old, lat_old, lon_old, lat_limits, lon_limits)

    # Plot an animation of new and old grids
    data_values = [data_new, data_old_crop]
    titles = ["New Lat Lon Grid", "Old Grid"]


    animated_time_series(data_values, time = time, 
                        main_title = main_title, titles = titles, y_labels = None, x_labels = None, c_labels = None, 
                        vmin = None, vmax = None, cmap = 'viridis', interval = 200, 
                        save_path = save_path)

    return

if __name__ == "__main__":
    main()
