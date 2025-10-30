import numpy as np
import os

PATH_SOURCE = "/home/jbassham/jack/data/ece228/year"
PATH_DEST = "/home/jbassham/jack/data/ece228/year/inputs"

START_YEAR = 2019
END_YEAR = 2020
HEM = 'nh'

def main():
    
    # Extract variables

    fnam = f"motion_ppv4_latlon_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    ui = data['u'] # zonal ice velocity
    vi = data['v'] # meridional ice velocity
    r = data['error']
    lat = data['lat']
    lon = data['lon']
    time = data['time']

    fnam = f"con_nimbus7_latlon_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    ic = data['ic']# zonal ice velocity

    fnam = f"wind_JRA55_latlon_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    uw = data['u']
    vw = data['v']

    print('Variable Files Loaded')

    # Mask ice concentration
    ic_raw = np.round(ic * 250) # raw value ice concentration (NSIDC)

    # NSIDC Masks 
    # 251 pole hole
    # 252 unused data
    # 253 coastline
    # 254 land
    ic = np.where((ic_raw == 251) | (ic_raw == 252) | (ic_raw == 253) | (ic_raw == 254), np.nan, ic)

    # Shift present day parameters forward one day, for one point Middle Weddell
    uit = ui[1:,:,:]
    vit = vi[1:,:,:]
    uwt = uw[1:,:,:]
    vwt = vw[1:,:,:]
    ict = ic[1:,:,:]
    tt = time[1:]
    rt = r[1:,:,:]

    # Remove last day from previous day parameters
    icy = ic[:-1,:,:]
    ty = time[:-1]

    # Create list of input variables
    invars = [uit, vit, rt, uwt, vwt, icy]

    # Mask spatial indices with ice error greater than 999 and 0 concentration ice motion
    mask = (ict == 0) | (np.isnan(ict)) | (rt >= 999) | (rt <= -999)

    # NaN out points meeting mask condition
    invars = [np.where(mask, np.nan, var) for var in invars]

    print("Variables Masked")

    # Compute temporal mean of inputs at every gridpoint
    grid_means = [np.nanmean(var, axis = 0) for var in invars]

    # Remove temporal mean from inputs at every gridpoint
    invars_demean = [var - mean for var, mean in zip(invars, grid_means)]

    # Compute global standard deviation (entire time series) of demeaned inputs
    stds = [np.nanstd(var) for var in invars_demean]

    # Normalize demeaned inputs by global standard deviation (z-score normalization)
    invars_norm = [var / std for var, std in zip(invars_demean, stds)]

    # Save normalized input variables
    uitn, vitn, rtn, uwtn, vwtn, icyn = invars_norm

    fnam = f'inputs_normalized_{HEM}_{START_YEAR}_{END_YEAR}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        uitn = uitn, vitn = vitn, 
        rtn = rtn, 
        uwtn = uwtn, vwtn = vwtn,
        icyn = icyn
        )
    
    print(f"Normalized inputs saved at: \n {PATH_DEST}/{fnam}")

    # Save statistics for normalizing
    uit_m, vit_m, rt_m, uwt_m, vwt_m, icy_m = grid_means
    uit_std, vit_std, rt_std, uwt_std, vwt_std, icy_std = stds

    fnam = f'stats_for_normalization_{HEM}_{START_YEAR}_{END_YEAR}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        uit_m = uit_m, vit_m = vit_m, 
        rt_m = rt_m, 
        uwt_m = uwt_m, vwt_m = vwt_m, 
        icy_m = icy_m,
        uit_std = uit_std, vit_std = vit_std, 
        rt_std = rt_std, 
        uwt_std = uwt_std, vwt_std = vwt_std, 
        icy_std = icy_std
    )

    print(f"Stats for normalizing saved at: \n {PATH_DEST}/{fnam}")

    # Save dates

    fnam = f'time_today_{HEM}_{START_YEAR}_{END_YEAR}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        time_today = tt
    )

    print(f"'Present day' time saved at: \n {PATH_DEST}/{fnam}")
    
    fnam = f'lat_lon_{HEM}_{START_YEAR}_{END_YEAR}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        lat = lat,
        lon = lon
    )

    print(f"Lat Lon vars saved at: \n {PATH_DEST}/{fnam}")

    return

if __name__ == "__main__":
    main()
