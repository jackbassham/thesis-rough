import gc
import numpy as np
import os

from .path import (
    PATH_SOURCE,
    PATH_DEST,
    FSTR_END_IN,
    FSTR_END_OUT
)

def main():

    # Load in variables
    fnam = f"motion_ppv4_latlon_{FSTR_END_IN}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)

    ui = data['u'] # zonal ice velocity
    vi = data['v'] # meridional ice velocity
    ri = data['r'] # ice velocity uncertainty (same for u and v)

    print('Ice Velocity, Uncertainty Loaded')

    fnam = f"con_nimbus7_latlon_{FSTR_END_IN}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)

    ci = data['ci']# ice concentration

    print('Concentration Loaded')

    fnam = f"wind_jra55_latlon_{FSTR_END_IN}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)

    ua = data['u']
    va = data['v']

    print('Wind Loaded')

    # Delete 'data' from memory
    del data
    gc.collect()

    print('Variable Files Loaded')
    print('')

    # Mask ice concentration based on NSIDC dataset mask values
    ci_raw = np.round(ci * 250) # raw value ice concentration (NSIDC)

    # NSIDC Mask values
    # 251 pole hole
    # 252 unused data
    # 253 coastline
    # 254 land
    ci = np.where((ci_raw == 251) | (ci_raw == 252) | (ci_raw == 253) | (ci_raw == 254), np.nan, ci)

    print('Raw concentration masked based on NSIDC masks.')

    nt, _, _ = np.shape(ci)

    # Assign threshold of percent ice free days for concentration masking
    thresh_ice_free = .80 * nt # 80% of days

    # Count number of days ice free at each spatial location
    n_ice_free = np.sum(ci == 0, axis = 0)

    # Create mask for ice free days above threshold
    mask_ice_free = n_ice_free > threshold

    # Mask concentration based on threshold
    ci = np.where(mask_ice_free, np.nan, ci)

    # Shift present day parameters forward one day, for one point Middle Weddell
    ui_t0 = ui[1:,:,:]
    vi_t0 = vi[1:,:,:]
    ua_t0 = ua[1:,:,:]
    va_t0 = va[1:,:,:]
    ri_t0 = ri[1:,:,:]

    # Get present day ice concentration for masking
    ci_t0 = ci[1:,:,:]

    # Remove last day from previous day parameters
    ci_t1 = ci[:-1,:,:]

    # Create list of input variables
    invars = [ui_t0, vi_t0, ri_t0, ua_t0, va_t0, ci_t1]

    # Compute variance of zonal and meridional ice velocities
    var_ui_t0 = np.nanvar(ui_t0, axis = 0)
    var_vi_t0 = np.nanvar(vi_t0, axis = 0)

    # Set minimum variance threshold
    thresh_ui_var = 0.05

    # Define mask where ice velocity variance is lower than threshold
    mask_var = (var_ui_t0 < thresh_ui_var) | (var_vi_t0 < thresh_ui_var)

    print(f'Mask defined where ice velocity variance < {var_thresh}')
    print('')

    # Define minimum ice concentration threshold
    thresh_ci = 0.15

    # Define mask where ice concneration less than .15 or nan
    mask_ci = (ci_t0 <= ci_thresh) | (np.isnan(ci_t0))

    # Combine masks
    total_mask = mask_var | mask_ci

    # NaN out points meeting mask condition
    invars_masked = [np.where(total_mask, np.nan, var) for var in invars]

    print(f'Mask defined where ice concentration values <= {ci_thresh} or nan.')
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
    
    # Save normalized input variables
    fnam = f'masked_normalized_{FSTR_END_OUT}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        ui_t0 = ui_norm, vi_t0 = vi_norm, 
        ri_t0 = ri_norm, 
        ua_t0 = ua_norm, va_t0 = va_norm,
        ci_t1 = ci_norm
        )

    print(f"Normalized inputs saved at: \n {PATH_DEST}/{fnam}")

    # Save statistics for normalization

    fnam = f'stats_norm_{FSTR_END_OUT}.npz'

    np.savez(
        os.path.join(PATH_DEST, fnam),
        ui_bar = ui_bar, vi_bar = vi_bar,
        ua_bar = ua_bar, va_bar = va_bar, 
        ci_bar = ci_bar,
        ua_std = ua_std, va_std = va_std, 
        ci_std = ci_std
    )

    print(f"Stats for normalizing saved at: \n {PATH_DEST}/{fnam}")

if __name__ == "__main__":
    main()
