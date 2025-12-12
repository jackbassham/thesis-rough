import numpy as np
import os
import sys

# Reads binary JRA-55 daily 3-Hourly near-surface (10m) wind vector data from Mazloff server and writes into .npz file
# Oringinal Data from: "https://rda.ucar.edu/datasets/d628000/"
# JRA55 README here: ""***""
#  ***credit source here***  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

# NOTE 1989 on Mazloff Server has 366 days not 365 (doesn't line up with leap year rule)
START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

LAT_LIMITS = [float(x) for x in os.getenv("LAT_LIMITS").split(",")] # South to North latitude bounds, degrees
LON_LIMITS = [float(x) for x in os.getenv("LON_LIMITS").split(",")] # West to East longitude bounds, degrees

TIMESTAMP_RAW = os.getenv("TIMESTAMP_RAW") # timestamp version for raw data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remaining global variables defined here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define file source names for u and v vector components 
FNAM_U = "jra55_u10m_{year}"
FNAM_V = "jra55_v10m_{year}"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define file source path (Mazloff Lab Servers)
PATH_SOURCE = "/project_shared/jra55/"

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define absolute data download destination path relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data',
        'raw', 
        HEM, 
        TIMESTAMP_RAW)
)

# Create the direectory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additional global variables here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_RAW}"

def main():

    # Create time array of dates for year range
    time = generate_datetime_array(START_YEAR, END_YEAR)

    # Create guassian latitude and longitude coordinate variables
    lat, lon = generate_guassian_coordinates()
    
    do_years = np.arange(START_YEAR, END_YEAR+1)

    # Initialize empty lists for total time series data
    u_total = [] # Zonal wind component
    v_total = [] # Meridional wind component

    for year in do_years:
        fnam_u = FNAM_U.format(year = year)
        fnam_v = FNAM_V.format(year = year)

        # Read gridded binary data into variables
        u = read_JRA55(year, fnam_u, lat, lon)
        v = read_JRA55(year, fnam_v, lat, lon)
        
        # Check that u and v successfully read
        if u is None:
            print(f"Error reading u for {year}, Quiting")
            sys.exit(1)
        
        if v is None:
            print(f"Error reading v for {year}, Quiting")
            sys.exit(1)

        # Append to total time series
        u_total.append(u)
        v_total.append(v)

        # Print success
        print(f"{year} u and v read")

    # Concatenate total u and v arrays along time dimension
    u_total = np.concatenate(u_total, axis = 0)
    v_total = np.concatenate(v_total, axis = 0)

    # Save time series data as npz variables
    fnam = f"wind_JRA55_gaussian_{FSTR_END_OUT}"

    np.savez_compressed(
        os.path.join(PATH_DEST, fnam), 
        u = u_total, 
        v = v_total, 
        time = time, 
        lat = lat, 
        lon = lon
    )
    
    print(f"Variables Saved at path {PATH_DEST}/{fnam}.npz")
    
    return


def generate_datetime_array(start_year, end_year):
    """Generates array of datetime64 objects input year range"""
    
    # Create list to hold dates
    dates_total = []
    
    do_years = np.arange(start_year, end_year + 1)

    for year in do_years:
        # Create array of dates for each year 
        # NOTE np.arange exclusive of last day
        dates = np.arange(f'{year}-01-01', f'{year+1}-01-01', dtype='datetime64[D]')
        # Append year's date to total date array
        dates_total.append(dates)
    
    dates_total = np.concatenate(dates_total, axis = 0)

    return dates_total
 

def generate_guassian_coordinates():
    """
    Generates a Guassian latitude longitude grid based on ~ 0.562° resolution used for JRA55 wind data 

    Global grid sized 640x320 points (0E to 359.438E and -89.57S to 89.57N)
    
    """

    # Construct array of latitude resolutions
    start_res = np.array([0.556914, 0.560202, 0.560946, 0.561227, 0.561363, 0.561440, 0.561487, 0.561518, 0.561539, 0.561554,
                         0.561566, 0.561575, 0.561582, 0.561587, 0.561592])

    mid_res = np.ones(289) * 0.561619268965519

    end_res = np.array([0.561592, 0.561587, 0.561582, 0.561575, 0.561566,
                       0.561554, 0.561539, 0.561518, 0.561487, 0.561440,
                       0.561363, 0.561227, 0.560946, 0.560202, 0.556914])

    dy = np.concatenate((start_res, mid_res, end_res))


    # Initialize latitude array (size 320)
    ny = len(dy)+1
    y = np.zeros(ny)
    y[0] = -89.57009 # Grid starting at 89.57 deg South

    # Construct the remaining latitude array from resolution above
    for i in range(1, ny):
        y[i] = y[i-1] + dy[i-1]

    # Initialize longitude array (size 640)
    nx = 640
    x = np.zeros(nx) # Grid starting at 0 deg East

    # Construct the remaining longitude array from resolution
    for i in range (1, nx):
        x[i] = x[i-1] + 0.5625

    return y, x


def read_JRA55(year, fnam, lat, lon):
    """
    
    Reads binary JRA-55 data in 3 hourly chunks and writes daily average to temp file object

    Option to crop to bounds with upper and lower lat and lon inputs
    
    """
    path = os.path.join(PATH_SOURCE, fnam)

    # Define number of days, handling condition for leap years
    if (year % 4 == 0) & (year % 100 != 0):
        days = 366
    elif (year % 4 == 0) & (year % 100 ==0) & (year % 400 == 0):
        days = 366
    else:
        days = 365
    
    try:
        with open(path, "rb") as file:
            time_steps_per_day = 8  # 24 hours / 3 hours per time step

            total_chunks = days * time_steps_per_day # Total number of 3 hourly chunks in yearly binary file

            # Read file into memory
            raw_data = np.frombuffer(file.read(), dtype = '>f') # '>f' big endian single floating point precission

            print(raw_data.size)
            
            try:
                # Check for improper binary file size
                if raw_data.size != total_chunks * len(lat) * len(lon):
                    raise ValueError(f"Data size mistmatch for {fnam}, expected {total_chunks * len(lat) * len(lon)}, got {raw_data.size}")
                    
                # Reshape raw data to 3 hourly chunks shaped (total_chunks, lat, lon)
                chunk_3hr_data = raw_data.reshape(total_chunks, len(lat), len(lon))
 
                # Reshape the arrays from [time, lat, lon] to [days_in_year, time_steps_per_day, lat, lon]
                data_reshaped = chunk_3hr_data.reshape(days, time_steps_per_day, len(lat), len(lon))
 
                # Compute the daily mean along the 3-hourly time steps dimension (axis 1)
                daily_data = np.mean(data_reshaped, axis=1)


            except ValueError as e:
                print(e)


            # # Reshape raw data to 3 hourly chunks shaped (total_chunks, lat, lon)
            # chunk_3hr_data = raw_data.reshape(total_chunks, len(lat), len(lon))

            # # Crop if bounds provided
            # if lat_lim is not None:
            #     # Find indices to crop coordinate variables to bounds
            #     ilat = np.where((lat >= lat_lim[0]) & (lat <= lat_lim[1]))[0]
            #     # Crop coordinate variable
            #     lat = lat[ilat]
            # else:
            #     ilat = np.arange(len(lat)) # indices same as original
            #     lat = lat

            # if lon_lim is not None:
            #     # Find indices to crop coordinate variables to bounds
            #     ilon = np.where((lon >= lon_lim[0]) & (lon <= lon_lim[1]))[0]
            #     # Crop coordinate variable
            #     lon = lon[ilon]
            # else:
            #     ilon = np.arange(len(lon)) # indices same as original
            #     lon  = lon


            # # Slice data to lat indices, then lon indices
            # cropped_3hr_data = chunk_3hr_data[:, ilat, :][:, :, ilon]

            # # Create daily data shaped in 8 3hourly chunks
            # daily_data = cropped_3hr_data.reshape(days, 8, len(ilat), len(ilon))

            # # Average the 8 3-hourly chunks for daily average
            # daily_data = daily_data.mean(axis = 1)

    except FileNotFoundError:
        # Continue to next year if file doesn't exist
        print(f"File '{fnam}' not found")

    except IOError as e:
        # Continue to next year after handling other errors
        print(f"Error reading file '{fnam}'")
        
    return daily_data



if __name__ == "__main__":
    main()
