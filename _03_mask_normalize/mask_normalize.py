import gc
from helpers import load_npz_data
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Annotated
from types import (
    Array3D, 
    Mask2D, 
    Mask3D
)

# FIXME pass/ silence numpy RuntimeWarning: Mean of empty slice

# FIXME move to configuration object
# Define loading keys for each variable
VARIABLE_CONFIG = {
    'ui': {'file': 'ice_vel', 'keys': ['ui']},
    'vi': {'file': 'ice_vel', 'keys': ['vi']},
    'ri': {'file': 'ice_vel', 'keys': ['ri']},
    'ua': {'file': 'wind', 'keys': ['ua']},
    'va': {'file': 'wind', 'keys': ['va']},
    'ci': {'file': 'ice_conc', 'keys': ['ci']},
}


def main(cfg):

    # Load regrid data source path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Initialize empty dict for filenames
    filenames = {}

    # Iterate through datastet dicts
    for name, ds in cfg.dataset_config.datasets().items():
        # Build filename for each regrid dataset
        filenames[name] = cfg.dataset_config.build_filename(ds, 'regrid')

    # Load data into data store variable
    data = load_all_variables(path_regrid, filenames)

    ui = data['ui']
    vi = data['vi']
    ri = data['ri']
    ua = data['ua']
    vi = data['va']
    ci = data['ci']

    # Shift variables to present day input parameters
    ui_t0, vi_t0, ri_t0 = present_day(ui), present_day(vi), present_day(ri)
    ua_t0, va_t0 = present_day(ua), present_day(va)
    ci_t0 = present_day(ci)

    # Shift variables to previous day input parameters
    ci_t1 = previous_day(ci)

    # Create masks for bad points and land/ open ocean
    mask_bad, mask_land_ocean = create_data_masks(
        ci_t0, ui_t0, vi_t0
    )

    



    # Take absolute value of uncertainty
    # NOTE negatives exist where data points are close to coastlines (NSIDC)
    # due to the possibility of False Ice, so that users can remove 
    # from dataset
    # NOTE
    # TODO see what abs value does, then consider removing entirely
    ri_t0 = np.abs(ri_t0)


    # Create list of input variables
    invars = [ui_t0, vi_t0, ri_t0, ua_t0, va_t0, ci_t1]

    # Load masked/normalized destination path
    path_mask_norm = cfg.path_config.data_stage_path('mask_norm')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_mask_norm) 

    # Define filename for mask
    filename = 'nan_mask.npz'

    # Save the mask
    np.savez(
        path_mask_norm / filename,
        nan_mask = nan_mask,
    )

    # Define land mask (just in case)
    land_mask = np.all(np.isnan(ci_t0), axis = 0)

    # Define filename for mask
    filename = 'land_mask.npz'

    # Save the mask
    np.savez(
        path_mask_norm / filename,
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
    filename = 'masked_normalized.npz'

    # Save the normalized data
    np.savez(
        path_mask_norm / filename,
        ui_t0 = ui_norm, vi_t0 = vi_norm, 
        ri_t0 = ri_norm, 
        ua_t0 = ua_norm, va_t0 = va_norm,
        ci_t1 = ci_norm
        )
    
    # Define file name for normalization statistics
    filename = 'stats_for_normalization.npz'

    # Save the normalization statistics
    np.savez(
        path_mask_norm / filename,
        ui_bar = ui_bar, vi_bar = vi_bar,
        ua_bar = ua_bar, va_bar = va_bar, 
        ci_bar = ci_bar,
        ua_std = ua_std, va_std = va_std, 
        ci_std = ci_std
    )


def load_all_variables(path_regrid: Path, filenames: dict[str: str]) -> dict[str: npt.NDArray]:
    """
    
    """

    # Initialize empty dict to store data
    data_store = {}

    # Iterate through variable configurations
    for var_config in VARIABLE_CONFIG.vaules():
        
        # Get key for variable
        file_key = var_config['file']

        # Load dataset file if it hasn't already
        if file_key not in data_store:
            data_store[file_key] = load_npz_data(path_regrid / filenames[file_key])

        # Iterate through dataset variable keys
        for key in var_config['keys']:
            # Load data variable into data store
            data_store[key] = data_store[file_key][key]

        return data_store


def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[-1:,:,:]


def create_data_masks(
        ci_t0: Array3D, ui_t0: Array3D, vi_t0: Array3D,
        perc_ice_free_threshold: float=0.70,
        ice_conc_threshold: float=0.15
) -> tuple[Mask3D, Mask2D]:
    """
    NOTE NSIDC considers up to 0.15 ice concentration 'ice free' for ice motion dataset
    """

    # Get NSIDC pre-normalization raw ice conentration
    ci_t0_raw = np.round(ci_t0 * 250)

    # List NSIDC flag values
    nsidc_flags = [
        251, # pole hole
        252, # unused data
        253, # coastline
        254, # land
    ]

    # Mask concentration based on NSIDC flag values
    ci_t0 = np.where(
        np.isin(ci_t0_raw, nsidc_flags),
        np.nan,
        ci_t0
    )

    # Get the number of days from ice concentration
    n_days = ci_t0.shape[0]

    # Count number of days ice free at each gridpoint
    n_ice_free = np.sum(ci_t0 <= ice_conc_threshold, axis = 0)

    # Create mask of nan values at bad data points
    mask_bad = (
        np.isnan(ci_t0)
        | np.isnan(ui_t0)
        | np.isnan(vi_t0)
        | (ci_t0 <= ice_conc_threshold)
        | (n_ice_free > (perc_ice_free_threshold * n_days))
    )

    # Define land/ open ocean mask, assuming these points always nan
    mask_land_ocean = np.all(np.isnan(ci_t0), axis = 0)

    return mask_bad, mask_land_ocean






if __name__ == "__main__":
    # NOTE remember this block is for direct script execution
    # know this because __name__ called is "__main__" (script!)
    # So when running script directly, we import load_config separately,
    # otherwise, when importing the module here, EVEN 'main()' 
    # in the case of master 'run_pipeline.py', this block is ignored
    # and the universal cfg is passed to main in 'run_pipeline.py'
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)
