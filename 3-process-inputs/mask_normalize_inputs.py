import gc
import numpy as np
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

LAT_LIMITS = [float(x) for x in os.getenv("LAT_LIMITS").split(",")] # South to North latitude bounds, degrees
LON_LIMITS = [float(x) for x in os.getenv("LON_LIMITS").split(",")] # West to East longitude bounds, degrees

RESOLUTION = int(os.getenv("RESOLUTION")) # Grid resolution, km

TIMESTAMP_IN = os.getenv("TIMESTAMP_IN") # timestamp version of input data

TIMESTAMP_OUT = os.getenv("TIMESTAMP_OUT") # timestamp version of inputs processed here

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additonal global variables here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}"
FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define absolute raw data directory source path relative to current
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        HEM, 
        'regrid')
)

# Define masked normalized data destination path relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        HEM, 
        'mask-norm')
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)


def main():


    # Extract variables
    fnam = f"motion_ppv4_latlon_{FSTR_END_IN}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    ui = data['u'] # zonal ice velocity
    vi = data['v'] # meridional ice velocity
    r = data['error'] # ice velocity uncertainty (same for u and v)
    lat = data['lat'] 
    lon = data['lon']
    time = data['time']

    print('Ice Velocity, Uncertainty Loaded')

    fnam = f"con_nimbus7_latlon_{FSTR_END_IN}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    ci = data['ci']# ice concentration

    print('Concentration Loaded')

    fnam = f"wind_JRA55_latlon_{FSTR_END_IN}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    ua = data['u']
    va = data['v']

    print('Wind Loaded')

    # Delete 'data' from memory
    del data
    gc.collect()

    print('Variable Files Loaded')
    print('')

    # Mask ice concentration
    ci_raw = np.round(ci * 250) # raw value ice concentration (NSIDC)

    # NSIDC Masks 
    # 251 pole hole
    # 252 unused data
    # 253 coastline
    # 254 land
    ci = np.where((ci_raw == 251) | (ci_raw == 252) | (ci_raw == 253) | (ci_raw == 254), np.nan, ci)

    print('Raw concentration masked based on NSIDC masks.')

    # Delete unused from memory
    del ui, vi, ua, va, ci, time, r
    gc.collect()

    print("Data shifted for 'present' and 'previous' days")

    # Create list of input variables
    invars = [uit, vit, rt, uat, vat, ciy]

    # Mask spatial indices with concentration less than .15, NaN concentration
    # NOTE keeping flag values for ice velocity uncertainties
    mask = (cit <= .15) | (np.isnan(cit))

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
    uit_masked, vit_masked, rt_masked, uat_masked, vat_masked, ciy_masked = invars_masked

    # Delete unused arrays from memory
    del invars
    del uit, vit, uat, vat, cit, ciy
    gc.collect()

    # Unpack statistics
    uit_bar, vit_bar, _, uat_bar, vat_bar, ciy_bar = grid_means

    _, _, _, uat_std, vat_std, ciy_std = global_stds

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
    uatn = (uat_masked - uat_bar) / uat_std

    vatn = (vat_masked - vat_bar) / vat_std

    ciyn = (ciy_masked - ciy_bar) / ciy_std

    print("'uat_bar', 'vat_bar', and 'ciy_bar' normalized by respective standard devations:")
    print(f"   {uat_std:.3f} cm/s, {vat_std:.3f} cm/s, {ciy_std:.3f}")
    print('')
    
    # Save normalized input variables
    fnam = f'inputs_normalized_{FSTR_END_OUT}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        uitn = uitn, vitn = vitn, 
        rtn = rtn, 
        uatn = uatn, vatn = vatn,
        ciyn = ciyn
        )

    print(f"Normalized inputs saved at: \n {PATH_DEST}/{fnam}")

    # Save statistics for normalization

    fnam = f'stats_for_normalization_{FSTR_END_OUT}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        uit_bar = uit_bar, vit_m = vit_bar,
        uat_bar = uat_bar, vat_m = vat_bar, 
        ciy_m = ciy_bar,
        cit_std = cit_std,
        uat_std = uat_std, vat_std = vat_std, 
        ciy_std = ciy_std
    )

    print(f"Stats for normalizing saved at: \n {PATH_DEST}/{fnam}")

    # Save dates

    fnam = f'time_today_{FSTR_END_OUT}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        time_today = tt
    )

    print(f"'Present day' time saved at: \n {PATH_DEST}/{fnam}")
    
    fnam = f'lat_lon_{FSTR_END_OUT}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        lat = lat,
        lon = lon
    )

    print(f"Lat Lon vars saved at: \n {PATH_DEST}/{fnam}")

    return

if __name__ == "__main__":
    main()
