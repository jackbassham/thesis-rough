import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# FIXME move to data download? So user is not downloading entire region

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

# FIXME Clean these up

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
