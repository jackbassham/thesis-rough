import io
import numpy as np
import os
import requests
import xarray as xr

from .param import (
    HEM, 
    START_YEAR,
    END_YEAR
)

from .path import PATH_DEST, FSTR_END_OUT

import temp_nasa_earth_data_file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global variables defined here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Enter valid base download URL (leaving off file from path) 
BASE_URL = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/{hem}/daily/"

# Enter valid file name (end of URL string)
END_URL = "icemotion_daily_{HEM}_25km_{year}0101_{year}1231_v4.1.nc"

def main():

    # Get abbreviated hemisphere for filename
    if HEM == "sh":
        hem = "south"
    else:
        hem = "north"

    # Format base url for hemisphere 
    base_url = BASE_URL.format(hem = hem)

    # Download lat and lon variables from one year
    fnam = END_URL.format(HEM = HEM, year = END_YEAR)

    # Concatenate entire url from base and filename
    url = base_url + fnam

    # Get lat and lon variables from one year
    temp = temp_nasa_earth_data_file(url)
    with xr.open_dataset(temp) as data:
        lat = data['latitude'] # EASE latitude shaped [y, x]
        lon = data['longitude'] # EASE longitude shaped [y, x]

    # Initialize lists for time series data
    u_total = []
    v_total = []
    r_total = []
    time_total = []

    # Define years to process (np.arrange() exclusive of last value)
    do_years = np.arange(START_YEAR, END_YEAR + 1)

    # Iterate through years
    for year in do_years:
        # Enter filename to download for each year in loop
        fnam = END_URL.format(year=year, HEM = HEM)

        url = base_url + fnam

        # Download file at year
        temp = temp_nasa_earth_data_file(url)
        with xr.open_dataset(temp) as data:
            u = data['u'].values                             # horizontal sea ice velocity [t, y, x], cm/s
            v = data['v'].values                             # vertical sea ice velocity [t, y, x], cm/s 
            r = data['icemotion_error_estimate'].values      # Ice motion uncertainty [t, y, x], cm/s
            time = data['time']  
            
        # Append year's data to list
        u_total.append(u)
        v_total.append(v)
        r_total.append(r)
        time_total.append(time)

        # Confirm download
        print(f"{year} downloaded")

    # Concatenate lists of data along time dimension
    u_total = np.concatenate(u_total, axis = 0)
    v_total = np.concatenate(v_total, axis = 0)
    r_total = np.concatenate(r_total, axis = 0)
    time_total = np.concatenate(time_total, axis = 0)

    # Convert time variable to numpy.datetime64 datatype
    time_total = np.array([np.datetime64(t) for t in time_total])

    # Save time series data as npz variables
    fnam = f"motion_ppv4_EASE_{FSTR_END_OUT}"
    path = os.path.join(PATH_DEST, fnam)

    np.savez_compressed(
        path, 
        u = u_total, 
        v = v_total, 
        r = r_total, 
        time = time_total, 
        lat = lat, 
        lon = lon
    )

    print(f"Variables Saved at path {path}.npz")

    return

if __name__ == "__main__":
    main()
