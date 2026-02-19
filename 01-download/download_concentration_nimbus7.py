from datetime import datetime, timedelta
import io
import numpy as np
import os
import requests
import time
import xarray as xr # With h5netcdf

from config.config import (
    HEM, 
    START_YEAR,
    END_YEAR
)

from config.path import PATH_RAW

from helpers.nasa_earth_data import get_temp_NED_file

# Downloads Daily NSIDC Sea Ice Concentrations (Nimbus7)
# from https://nsidc.org/data/nsidc-0051/versions/2
# Data saved as time series in .npz file
# ***credit source here***

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global variables defined here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List of possible sensor names for variables in files
VAR_NAMES = ['F08_ICECON', 'F11_ICECON', 'F13_ICECON', 'F17_ICECON']

# Enter valid base download URL (leaving off file from path) 
BASE_URL = "https://n5eil01u.ecs.nsidc.org/PM/NSIDC-0051.002/{date}/"

# Enter valid file name (end of URL string)
END_URL = "NSIDC0051_SEAICE_PS_{hem}25km_{date}_v2.0.nc"


def main():

    # Initialize log for missing days
    fnam = 'concentration_download_log.txt'
    path_log = os.path.join(PATH_RAW, fnam)

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
        if HEM == 'sh':
            hem = 'S'
        else:
            hem = 'N'

        fnam = END_URL.format(date = dstr_f, hem = hem)

        # Total url is parent and filename
        url = base_url+fnam
    
        # Get data in a temporary variable
        temp = get_temp_NED_file(url)

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
                        print(f'NO DATA FOR {date}, ICECON NaN')
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

            print(f'Error {date}')
            with open(path_log, "a") as log:
                log.write(f'{date}, NO DATA.\n')

        # Append ice concentration data to time series list
        ci_total.append(ci)

        print(f'{date} retrieved')

        # Continue to next day
        date += timedelta(days=1)

    # Concatenate concentration data along time dimension
    ci_total = np.concatenate(ci_total, axis = 0)

    # Define name for data file
    fnam = 'ice_conc_raw_nimbus7_ps.npz'

    # Create directory for the data if it doesn't already exist
    os.makedirs(PATH_RAW, exist_ok = True)

    # Save the data
    np.savez_compressed(
        os.path.join(PATH_RAW, fnam), 
        ci = ci_total, 
        time = time_total, 
        var_names = var_names_total, 
        allow_pickle = True
    )

    print(f'Raw ice concentration saved at {PATH_RAW}/{fnam}')

    return


if __name__ == "__main__":
    main()
