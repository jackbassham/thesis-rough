import gc
import numpy as np
import os

from _00_config.path import (
    PATH_REGRID,
    PATH_MASK_NORM,
)

def main():

    # Create the destination directory if it doesn't already exist
    os.makedirs(PATH_MASK_NORM, exist_ok = True)

    # Load in variables
    fnam = 'ice_vel_regrid_ppv4.npz'
    data = np.load(os.path.join(PATH_REGRID, fnam), allow_pickle=True)

    ui = data['u'] # zonal ice velocity
    vi = data['v'] # meridional ice velocity
    ri = data['r'] # ice velocity uncertainty (same for u and v)

    print('Ice Velocity, Uncertainty Loaded')

    fnam = 'ice_conc_regrid_nimbus7.npz'
    data = np.load(os.path.join(PATH_REGRID, fnam), allow_pickle=True)

    ci = data['ci']# ice concentration

    print('Concentration Loaded')

    fnam = 'wind_regrid_jra55.npz'
    data = np.load(os.path.join(PATH_REGRID, fnam), allow_pickle=True)

    ua = data['u']
    va = data['v']

    print('Wind Loaded')

    # Delete 'data' from memory
    del data
    gc.collect()

    # Mask ice concentration based on NSIDC dataset mask values
    ci_raw = np.round(ci * 250) # raw value ice concentration (NSIDC)

    # NSIDC Mask values
    # 251 pole hole
    # 252 unused data
    # 253 coastline
    # 254 land
    ci = np.where((ci_raw == 251) | (ci_raw == 252) | (ci_raw == 253) | (ci_raw == 254), np.nan, ci)

    print('Raw concentration masked based on NSIDC masks.')

    # Shift present day parameters forward one day
    ui_t0 = ui[1:,:,:]
    vi_t0 = vi[1:,:,:]
    ua_t0 = ua[1:,:,:]
    va_t0 = va[1:,:,:]
    ri_t0 = ri[1:,:,:]

    # Take absolute value of uncertainty
    # NOTE negatives exist where data points are close to coastlines (NSIDC)
    # due to the possibility of False Ice, so that users can remove 
    # from dataset
    # NOTE
    # TODO see what abs value does, then consider removing entirely
    ri_t0 = np.abs(ri_t0)


    # Get present day ice concentration for masking
    ci_t0 = ci[1:,:,:]

    # Remove last day from previous day parameters
    ci_t1 = ci[:-1,:,:]

    # Create list of input variables
    invars = [ui_t0, vi_t0, ri_t0, ua_t0, va_t0, ci_t1]

    # Get number of days in concentration variable
    nt, _, _ = np.shape(ci_t0)

    # Assign threshold for number of ice free days at a spatial gridpoint
    thresh_ice_free = .70 * nt # 70% of days

    # Count number of ice free days at each spatial gridpoint
    # NOTE NSIDC considers up to 0.15 ice concentration 'ice free' for ice motion dataset
    n_ice_free = np.sum(ci_t0 <= 0.15, axis = 0)

    # Define threshold for ice concentration at single point
    ci_thresh = 0.15

    # Create mask at spatial gridpoints where ice free days excede threshold
    # and at single data points where concentration below threshold
    # and where concentration or velocity is nan

    nan_mask = (np.isnan(ci_t0)) | (np.isnan(ui_t0)) | (np.isnan(vi_t0)) | (ci_t0 <= ci_thresh) | (n_ice_free > thresh_ice_free)

    print(f'Mask defined at gridpoints where "ice free" >= {thresh_ice_free} days')
    print(f'and where ice concentration values <= {ci_thresh} (ice edge)')

    # Define filename for mask
    fnam = 'nan_mask.npz'

    # Save the mask
    np.savez(
        os.path.join(PATH_MASK_NORM, fnam),
        nan_mask = nan_mask,
    )

    # Define land mask (just in case)
    land_mask = np.all(np.isnan(ci_t0), axis = 0)

    # Define filename for mask
    fnam = 'land_mask.npz'

    # Save the mask
    np.savez(
        os.path.join(PATH_MASK_NORM, fnam),
        land_mask = land_mask,
    )

    # NaN out points meeting mask condition
    invars_masked = [np.where(nan_mask, np.nan, var) for var in invars]

    print('Mask defined where ci is nan')
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
    ui_masked, vi_masked, ri_masked, ua_masked, va_masked, ci_masked = invars_masked

    # Delete unused arrays from memory
    del invars
    del ui, vi, ri, ua, va, ci
    gc.collect()

    # Unpack statistics
    ui_bar, vi_bar, _, ua_bar, va_bar, ci_bar = grid_means

    _, _, _, ua_std, va_std, ci_std = global_stds

    # Delete unused arrays from memory
    del _
    gc.collect()

    # Calculate speed
    Ui = np.sqrt(ui_masked ** 2 + vi_masked ** 2)

    # Get standard deviation of speed for normalization
    Ui_std = np.nanstd(Ui)

    # Delete unused arrays from memory
    del invars_masked, grid_means, global_stds
    gc.collect()

    # Normalize ice velocity and uncertainty by ice speed global standard deviation 
    # (z-score normalization)

    ui_norm = (ui_masked - ui_bar) / Ui_std
    vi_norm = (vi_masked - vi_bar) / Ui_std

    print("'uit_bar', 'vit_bar' normalized by 'cit_std:'")
    print(f"   {Ui_std:.3f} cm/s")
    print('')

    # Normalize uncertainty by standard deviation of speed
    ri_norm = ri_masked / Ui_std

    print(f"'rt' scaled by {Ui_std:.3f} cm/s:")
    print('')

    # Normalize remaning variables
    ua_norm = (ua_masked - ua_bar) / ua_std

    va_norm = (va_masked - va_bar) / va_std

    ci_norm = (ci_masked - ci_bar) / ci_std

    print("'ua', 'va', and 'ci' normalized by respective standard devations:")
    print(f"   {ua_std:.3f} cm/s, {va_std:.3f} cm/s, {ci_std:.3f}")
    print('')

    # Pack normalized input variables into list
    invars_norm = [ui_norm, vi_norm, ri_norm, ua_norm, va_norm, ci_norm]

    # Count number of data points in each variable
    total_points = [var.size for var in invars_norm]

    # Count the number of nans in each variable
    total_nan = [np.isnan(var).sum() for var in invars_norm]

    for p, n in zip(total_points, total_nan):
        print(f"total points/ total nan: {p} / {n}")
        print(f"num valid points {p - n}")
        print(f"frac nan (invalid) {n / p}")
    
    # Define data file name for normalized data
    fnam = 'masked_normalized.npz'

    # Save the normalized data
    np.savez(
        os.path.join(PATH_MASK_NORM, fnam),
        ui_t0 = ui_norm, vi_t0 = vi_norm, 
        ri_t0 = ri_norm, 
        ua_t0 = ua_norm, va_t0 = va_norm,
        ci_t1 = ci_norm
        )

    print(f"Normalized inputs saved at: \n {PATH_MASK_NORM}/{fnam}")

    # Define file name for normalization statistics
    fnam = 'stats_for_normalization.npz'

    # Save the normalization statistics
    np.savez(
        os.path.join(PATH_MASK_NORM, fnam),
        ui_bar = ui_bar, vi_bar = vi_bar,
        ua_bar = ua_bar, va_bar = va_bar, 
        ci_bar = ci_bar,
        ua_std = ua_std, va_std = va_std, 
        ci_std = ci_std
    )

    print(f"Stats for normalizing saved at {PATH_MASK_NORM}/{fnam}")


if __name__ == "__main__":
    main()
