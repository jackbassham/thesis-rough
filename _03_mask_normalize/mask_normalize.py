import helpers
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path

# TODO figure out project - specific types
# from types import (
#     Array3D, 
#     Mask2D, 
#     Mask3D
# )

# FIXME pass/ silence numpy RuntimeWarning: Mean of empty slice

# FIXME verbose and lots of logic in main, dealing with memory 
# usage vs modularity/ readability

def main(cfg):

    # Load regrid data source path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Load masked/normalized destination path
    path_mask_norm = cfg.path_config.data_stage_path('mask_norm')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_mask_norm) 
    
    # Initialize empty dict for dataset filenames
    filenames = {}

    # Iterate through datastet dicts
    for name, ds in cfg.dataset_config.datasets().items():
        # Build filename for each regrid source dataset
        filenames[name] = cfg.dataset_config.build_filename(ds, 'regrid')

    # Load in data variables
    ui, vi, ri = helpers.load_ice_vel(path_regrid, filenames['ice_vel'])
    ua, va = helpers.load_wind(path_regrid, filenames['wind'])
    ci = helpers.load_ice_conc(path_regrid, filenames['ice_conc'])

    # Shift variables to create present day input parameters
    ui_t0, vi_t0, ri_t0 = present_day(ui), present_day(vi), present_day(ri)
    ua_t0, va_t0 = present_day(ua), present_day(va)
    ci_t0 = present_day(ci)

    plt.pcolormesh(ui_t0[0])
    plt.title('ui_t0')
    plt.savefig('1debug_plot.png')

    # Shift variables to create previous day input parameters
    ci_t1 = previous_day(ci)

    # Create masks for bad points and land/ open ocean
    mask_bad, mask_land_ocean = create_data_masks(
        ci_t0, ui_t0, vi_t0
    )

    plt.pcolormesh(mask_bad[0])
    plt.title('mask_bad')
    plt.savefig('2debug_plot.png')

    # Save masks
    np.savez(
        path_mask_norm / 'masks.npz', 
        mask_bad = mask_bad, 
        mask_land_ocean = mask_land_ocean,
             )

    # Create dict of input parameters
    inputs = {
        'ui_t0': ui_t0, 'vi_t0': vi_t0, 'ri_t0': ri_t0,
        'ua_t0': ua_t0, 'va_t0': va_t0,
        'ci_t1': ci_t1,
    }

    plt.pcolormesh(inputs['ui_t0'][0])
    plt.title('inputs')
    plt.savefig('3debug_plot.png')

    # Mask bad points to nan (in place, no copy made)
    mask_inputs(inputs, mask_bad)

    plt.pcolormesh(inputs['ui_t0'][0])
    plt.title('masked inputs')
    plt.savefig('4debug_plot.png')

    # Compute the gridwise temporal mean of each input
    gridwise_means = compute_gridwise_means(inputs)

    plt.pcolormesh(gridwise_means['ui_t0'])
    plt.title('gridwise_means')
    plt.savefig('5debug_plot.png')

    # Compute the global standard deviations of each input
    global_stds = compute_global_stds(inputs)

    print('global_stds:')
    print(**global_stds)
    print()

    # Perform Z-score normalization of inputs, add ice speed std to dict
    normalized, global_stds = z_score_normalize_inputs(
        inputs, gridwise_means, global_stds,
    )
    
    # Save the normalized data
    save_arrays(path_mask_norm / 'masked_normalized.npz', normalized)

    # Save the gridwise means
    save_arrays(path_mask_norm / 'gridwise_means.nps', gridwise_means)

    # Save the global standard deviations
    save_arrays(path_mask_norm / 'global_stds.npz', global_stds)



def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[:-1,:,:]


def create_data_masks(
        ci_t0: npt.NDArray[np.floating], ui_t0: npt.NDArray[np.floating], vi_t0: npt.NDArray[np.floating],
        perc_ice_free_threshold: float=0.70,
        ice_conc_threshold: float=0.15
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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


def mask_inputs(inputs: dict, mask: npt.NDArray[np.floating]):
    """
    
    """
    
    # Iterate through inputs
    for value in inputs.values():
        # Fill masked points with nan in place
        np.putmask(value, mask, np.nan)


def compute_gridwise_means(
        inputs: dict[str, npt.NDArray[np.floating]], axis: int=0
    ) -> dict[str, npt.NDArray[np.floating]]:
    """
    
    """
    # Initialize empty dict
    gridwise_means = {}

    # Iterate through inputs
    for name, array in inputs.items():
        # Compute gridwise temporal means
        gridwise_means[name] = np.nanmean(array, axis = axis)

    return gridwise_means


def compute_global_stds(
        inputs: dict[str, npt.NDArray[np.floating]]
    ) -> dict[str, npt.NDArray[np.floating]]:
    """
    
    """
    
    # Initialize empty dict
    global_stds = {}

    # Iterate through inputs
    for name, array in inputs.items():
        # Compute global standard deviations
        global_stds[name] = np.nanstd(array)

    return global_stds


def z_score_normalize_inputs(
        inputs: dict[str, npt.NDArray[np.floating]],
        gridwise_means: dict[str, npt.NDArray[np.floating]],
        global_stds: dict[str, npt.NDArray[np.floating]],
    ) -> tuple[dict[str, npt.NDArray[np.floating]], dict[str, npt.NDArray[np.floating]]]:
    """
    # NOTE: Normalization (z-score, for comparison between variables - 0 mean, 1 std)
    # 1. Compute temporal mean, gridwise
    # 2. Compute global standard deviation
    # 3. Remove mean and divde by standard deviation
    # 4. ** Ice velocities here are normalized by the standard deviation of the speed
    # 5. ** Uncertainty here is scaled by speed
    """

    # Initialize dict for normalized inputs
    normalized = {}

    print(f'zscore_in {inputs['ui_t0'][0]}')

    # Get standard deviation of ice speed for noralization
    Ui_t0_std = np.nanstd(
        np.sqrt(inputs['ui_t0']**2 + inputs['vi_t0']**2)
    )

    print(f'Ui_t0_std {Ui_t0_std}')

    # Add to dict for saving
    global_stds['Ui_t0'] = Ui_t0_std

    # Ice Velocity: remove each point's mean and nornalize by std dev speed
    normalized['ui_t0'] = (inputs['ui_t0'] - gridwise_means['ui_t0']) / Ui_t0_std
    normalized['vi_t0'] = (inputs['vi_t0'] - gridwise_means['vi_t0']) / Ui_t0_std

    # Uncertainty: take absolute value and normalize by std ice speed
    normalized['ri_t0'] = np.abs(inputs['ri_t0']) / Ui_t0_std

    # Wind: remove each point's mean and normalize by respective std dev
    normalized['ua_t0'] = (inputs['ua_t0'] - gridwise_means['ua_t0']) / global_stds['ua_t0']
    normalized['va_t0'] = (inputs['va_t0'] - gridwise_means['va_t0']) / global_stds['va_t0']

    # Ice Concentration: remove each point's mean and normalize by respective std dev
    normalized['ci_t1'] = (inputs['ci_t1'] - gridwise_means['ci_t1']) / global_stds['ci_t1']

    return normalized, global_stds


def save_arrays(path: Path, filename: str, arrays: dict[str, npt.NDArray[np.floating]]) -> None:
    """
    
    """

    # Create path if it doesn't already exist
    path.mkdir(partents=True, exist_ok=True)

    # Save all key (varable name), value pairs (array)
    np.savez(path / filename, **arrays)


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
