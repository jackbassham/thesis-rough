import gc
import numpy as np
import os

#########################################################
# NOTE Adapted from ProcessInputsUncertaintyV5.ipynb with changes;
# 1. KEEP FLAG VALUES in uncertainty!
#########################################################

PATH_SOURCE = "/home/jbassham/jack/data/sh"
PATH_DEST = "/home/jbassham/jack/data/sh/inputs_v6"

START_YEAR = 1992
END_YEAR = 2020
HEM = 'sh'

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

    print('Ice Velocity, Uncertainty Loaded')

    fnam = f"con_nimbus7_latlon_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    ic = data['ic']# zonal ice velocity

    print('Concentration Loaded')

    fnam = f"wind_JRA55_latlon_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    uw = data['u']
    vw = data['v']

    print('Wind Loaded')

    # Delete 'data' from memory
    del data
    gc.collect()

    print('Variable Files Loaded')
    print('')


    # Mask ice concentration
    ic_raw = np.round(ic * 250) # raw value ice concentration (NSIDC)

    # NSIDC Masks 
    # 251 pole hole
    # 252 unused data
    # 253 coastline
    # 254 land
    ic = np.where((ic_raw == 251) | (ic_raw == 252) | (ic_raw == 253) | (ic_raw == 254), np.nan, ic)

    print('Raw concentration masked based on NSIDC masks.')

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

    # Delete unused from memory
    del ui, vi, uw, vw, ic, time, r
    gc.collect()

    print("Data shifted for 'present' and 'previous' days")

    # Create list of input variables
    invars = [uit, vit, rt, uwt, vwt, icy]

    # Mask spatial indices with concentration less than .15, NaN concentration
    # NOTE keeping flag values for ice velocity uncertainties
    mask = (ict <= .15) | (np.isnan(ict))

    # NaN out points meeting mask condition
    invars_masked = [np.where(mask, np.nan, var) for var in invars]

    print('Inputs masked where ice concentration values <= .15 or nan.')
    print('')

    # NOTE: Normalization (z-score, for comparison between variables - 0 mean, 1 std)
    # 1. Compute temporal mean, gridwise
    # 2. Compute global standard deviation
    # 3. Remove mean and divde by standard deviation
    # 4. ** Ice velocities here are normalized by the standard deviation of the speed
    # 5. ** Uncertainty here is scaled by ci_std

    # Compute temporal mean of inputs at every gridpoint
    grid_means = [np.nanmean(var, axis = 0) for var in invars_masked]

    # Compute global stds
    global_stds = [np.nanstd(var) for var in invars_masked]

    # Unpacked masked variables
    uit_masked, vit_masked, rt_masked, uwt_masked, vwt_masked, icy_masked = invars_masked

    # Delete unused arrays from memory
    del invars
    del uit, vit, uwt, vwt, ict, icy
    gc.collect()

    # Unpack statistics
    uit_bar, vit_bar, _, uwt_bar, vwt_bar, icy_bar = grid_means

    _, _, _, uwt_std, vwt_std, icy_std = global_stds

    # Delete unused arrays from memory
    del _
    gc.collect()

    # Calculate speed
    cit = np.sqrt(uit_masked ** 2 + vit_masked ** 2)

    # Get standard deviation of speed for normalization
    cit_std = np.nanstd(cit)

    # Delete unused arrays from memory
    del invars_masked, grid_means, global_stds
    gc.collect()

    # Normalize ice velocity and uncertainty by ice speed global standard deviation 
    # (z-score normalization)

    uitn = (uit_masked - uit_bar) / cit_std
    vitn = (vit_masked - vit_bar) / cit_std

    print("'uit_bar', 'vit_bar' normalized by 'cit_std:'")
    print(f"   {cit_std:.3f} cm/s")
    print('')

    # Normalize uncertainty by standard deviation of speed
    rtn = rt_masked / cit_std

    print(f"'rt' scaled by {cit_std:.3f} cm/s:")
    print('')

    # Normalize remaning variables
    uwtn = (uwt_masked - uwt_bar) / uwt_std

    vwtn = (vwt_masked - vwt_bar) / vwt_std

    icyn = (icy_masked - icy_bar) / icy_std

    print("'uwt_bar', 'vwt_bar', and 'icy_bar' normalized by respective standard devations:")
    print(f"   {uwt_std:.3f} cm/s, {vwt_std:.3f} cm/s, {icy_std:.3f}")
    print('')

    # Create destination path if it doesn't exist
    os.makedirs(PATH_DEST, exist_ok=True)
    
    # Save normalized input variables
    fnam = f'inputs_normalized_{HEM}_{START_YEAR}_{END_YEAR}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        uitn = uitn, vitn = vitn, 
        rtn = rtn, 
        uwtn = uwtn, vwtn = vwtn,
        icyn = icyn
        )

    print(f"Normalized inputs saved at: \n {PATH_DEST}/{fnam}")

    # Save statistics for normalization

    fnam = f'stats_for_normalization_{HEM}_{START_YEAR}_{END_YEAR}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        uit_bar = uit_bar, vit_m = vit_bar,
        uwt_bar = uwt_bar, vwt_m = vwt_bar, 
        icy_m = icy_bar,
        cit_std = cit_std,
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
