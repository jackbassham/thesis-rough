from datetime import datetime, timedelta
import io
import numpy as np
import os
import requests
import time
import xarray as xr # With h5netcdf

# TODO fnam move grid to end of name

# Downloads Daily NSIDC Sea Ice Concentrations (Nimbus7)
# from https://nsidc.org/data/nsidc-0051/versions/2
# Data saved as time series in .npz file
# ***credit source here***

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

# Nasa Earthdata login credentials for download
USER = os.getenv("USER") # username
PASS = os.getenv("PASS") # password

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remaining global variables defined here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List of possible sensor names for variables in files
VAR_NAMES = ['F08_ICECON', 'F11_ICECON', 'F13_ICECON', 'F17_ICECON']

# Enter valid base download URL (leaving off file from path) 
BASE_URL = "https://n5eil01u.ecs.nsidc.org/PM/NSIDC-0051.002/{date}/"

# Enter valid file name (end of URL string)
END_URL = "NSIDC0051_SEAICE_PS_{hem}25km_{date}_v2.0.nc"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define data absolute download destination path relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        HEM, 
        'raw')
)

# Create the direectory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)

def main():

    # Initialize log for missing days
    fnam_log = f'concentration_download_log_{HEM}{START_YEAR}{END_YEAR}.txt'
    path_log = os.path.join(PATH_DEST, fnam_log)

    # Define start and end dates for year(Test 2020) 
    start_date = datetime(START_YEAR, 1, 1)
    end_date = datetime(END_YEAR, 12, 31)

    # Initialize lists for time series variables
    ci_total = [] # Ice Concentration
    time_total = [] # Dates
    var_names_total = [] # Variable names (vary by day)

    # Initialize first day
    date = start_date

    # Loop through dates
    while date <= end_date:

        # Format date string for URL and filename
        dstr_url = date.strftime("%Y.%m.%d")
        dstr_f = date.strftime("%Y%m%d")
    
        # Declare parent directory and filename strings for each day
        base_url = BASE_URL.format(date = dstr_url)

        # Get abbreviated hemisphere for filename
        if HEM == "sh":
            hem = "S"
        else:
            hem = "N"

        fnam = END_URL.format(date = dstr_f, hem = hem)

        # Total url is parent and filename
        url = base_url+fnam
    
        # Get data in a temporary variable
        temp = temp_nasa_earth_data_file(url)

        # If retrieved from url sucessfully
        if temp is not None:
            # Store data from temp in xarray dataset
            with xr.open_dataset(temp) as ds:
                # Initialize variable name check
                found_name = False
                # Loop through variable names
                for name in VAR_NAMES:
                    # Extract data for varaiable name
                    if name in ds:
                        # Extract data for varibale name
                        ci = ds[name].values
                        # Append variable name to list
                        var_names_total.append(name)
        
                        # Exit loop when name found
                        found_name = True
                        break
                
                # If name is still not found
                if not found_name:
                        ci = np.nan
                        var_names_total.append(None)
                        print(f"NO DATA FOR {date}, ICECON NaN")
                        # Log missing data
                        with open(path_log, "a") as log:
                            log.write(f"{date}, NO DATA, ICECON NaN.\n")
                                # Append time variable for time series

                # Retrieve time variable and append to list
                time_total.append(ds['time'])

        # If unsuccessful set variable to nan for day
        else:
            ci = np.nan
            var_names_total.append(None)
            time_total.append(date)

            print(f"Error {date}")
            with open(path_log, "a") as log:
                log.write(f"{date}, NO DATA.\n")

        # Append ice concentration data to time series list
        ci_total.append(ci)

        print(f'{date} retrieved')

        # Continue to next day
        date += timedelta(days=1)

    # Concatenate concentration data along time dimension
    ci_total = np.concatenate(ci_total, axis = 0)

    # Save time series data as npz variables
    fnam = f"con_nimbus7_ps_{HEM}{START_YEAR}{END_YEAR}"
    np.savez_compressed(
        os.path.join(PATH_DEST, fnam), 
        ci = ci_total, 
        time = time_total, 
        var_names = var_names_total, 
        allow_pickle = True)

    print(f"Variables Saved at path {PATH_DEST + fnam}.npz")

    return


def temp_nasa_earth_data_file(url, retries = 3, delay = 15):
    """
    
    Gets temporary file from Nasa Earth Data Website via URL
    If connection fails, retries 3 times with 15 second delay

    """
    ### Create session for NASA Earthdata ###
    # Overriding requests.Session.rebuild_auth to mantain authentication headers when redirected
    # Custom subclass to extend functionality of parent class requests.session to maintain authentication headers
    # when server redirects requests
    class SessionWithHeaderRedirection(requests.Session):
        # Host for which authentication headers maintained
        AUTH_HOST = 'urs.earthdata.nasa.gov'
    
        # Define 'costructor method' for sublclass SessionWithHeaderRedirection  
        # Called to initialze attributes of subclass when created (here with parameters username and password)
        # 'self' parameter is in reference to the instance constructed (our subclass)
        def __init__(self, username, password):
            # Call constructor method for parent class (executing initialization code defined in parent class)
            super().__init__()
            # Initialize authentication atribute in class containing username and password
            self.auth = (username, password)

        # Overrides from the library to keep headers when redirected to or from the NASA auth host
        def rebuild_auth(self, prepared_request, response):
            headers = prepared_request.headers
            url = prepared_request.url

            if 'Authorization' in headers:
                original_parsed = requests.utils.urlparse(response.request.url)
                redirect_parsed = requests.utils.urlparse(url)

                if (original_parsed.hostname != redirect_parsed.hostname) and \
                        redirect_parsed.hostname != self.AUTH_HOST and \
                        original_parsed.hostname != self.AUTH_HOST:
                    del headers['Authorization']

    # Create session with the user credentials that will be used to authenticate access to the data
    session = SessionWithHeaderRedirection(USER, PASS)

    for attempt in range(retries):
        try:
            # submit the request using the session
            response = session.get(url, stream=True)
            # '200' means success
            StatusCode = response.status_code
            print(StatusCode)
            # raise an exception in case of http errors
            response.raise_for_status()  

            # Read response content to temp using BytesIO object
            temp = io.BytesIO(response.content)

            return temp

        except requests.exceptions.HTTPError as e:
            # Handle any errors here
            print(f"HTTP Error on attempt {attempt + 1}/{retries}: {e}")
            if response.status_code == 503: # Server unavailable
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("Max retries reached, skipping file.")
                    return None
    
        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("Max retires reached, skipping file.")
                return None


if __name__ == "__main__":
    main()
