import calendar
import numpy as np
from pathlib import Path
import sys


import _06_evaluate.metric_fcns
from analysis import (plot_fcns, helpers)



MODEL_STRS = [ 'ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd',]

DATA_ROOT = Path('/data/globus/jbassham/thesis-rough')

try:    
    model_idx = int(sys.argv[1]) 
except (IndexError, ValueError):
    model_idx = None

if model_idx is not None:
    MODEL_STR = MODEL_STRS[model_idx]
else:
    MODEL_STR = MODEL_STRS[0]

HEMISPHERE = 'south'

TIMESTAMP = '05222026_1652'

TIMESTAMP_MODEL_INPUTS = TIMESTAMP
TIMESTAMP_REGRID = TIMESTAMP

N_MEMBERS = 10

BASE_SOURCE_PATH = Path(
    DATA_ROOT
    / 'model-output'
    / MODEL_STR
    / HEMISPHERE
    / TIMESTAMP
)

BASE_DEST_PATH = Path(
    DATA_ROOT
    / 'analysis2'
    / 'metrics_monthly'
    / MODEL_STR
    / HEMISPHERE
    / TIMESTAMP
)


ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')


def main():

    # Define list of metric strings
    metric_strs = [
        'correlation',
        'weighted_correlation', 
        'skill', 
        'weighted_skill', 
        'rmse'
    ]

    # Load path to model inputs
    path_model_inputs = Path(DATA_ROOT / 'model_inputs' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS)

    # Load path to masked normalized data
    path_mask_norm = Path(DATA_ROOT / 'mask_norm' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS)

    # Load lists of masked predictions and true values from each member
    preds_list, trues_list, ri_t0s_list, test_indices = helpers.load_and_mask_member_preds(
        N_MEMBERS,
        BASE_SOURCE_PATH,
        path_model_inputs,
        path_mask_norm,
        return_indices=True,
    )

    print(f'~~~~~All member masked preds and trues loaded~~~~~')

    # Load time from cordinates file
    time_t0 = np.load(
        Path(DATA_ROOT / 'regrid' / HEMISPHERE / TIMESTAMP_REGRID)
        / 'coordinates.npz'
    )['time_t0']

    # Make destination path if it does not already exist
    BASE_DEST_PATH.mkdir(parents=True, exist_ok=True)

    
    # Compute metrics for each ensemble member
    for metric_str in metric_strs:

        print(f'~~~~~~~~~~~{metric_str.upper()}~~~~~~~~~~~~~~')

        metric_fcn = getattr(_06_evaluate.metric_fcns, metric_str)

        monthly_all_members = []

        # Initialize dict for extra uncertainty keyword argument for weighted metrics
        metric_kwargs = {}

        for m in range(N_MEMBERS):

            pred = preds_list[m]
            true = trues_list[m]

            if 'weighted' in metric_str:
                # Add current month's uncertainties array to kword arguments
                r_input = ri_t0s_list[m]

            else:
                r_input = None

            # Compute member memtric 
            monthly_metric = metric_fcn_month(
                pred,
                true,
                time_t0[test_indices[f'{m:02d}']],
                metric_fcn,
                ri_t0=r_input
            )  # expected shape: (channel, height, width)

            monthly_all_members.append(monthly_metric)

            print(f'Finished member {m} of {N_MEMBERS} for {metric_str}')

        monthly_all_members = np.stack(monthly_all_members, axis=0)
        # shape: (member, month, channel, height, width)

        np.savez(
            BASE_DEST_PATH / f'monthly_all_members_{metric_str}.npz',
            monthly_all_members=monthly_all_members,
        )

        metric_mean = np.nanmean(monthly_all_members, axis=0)

        if N_MEMBERS > 1:
            metric_sem = np.nanstd(monthly_all_members, axis=0) / np.sqrt(N_MEMBERS)
        else:
            metric_sem = None

        np.savez(
            BASE_DEST_PATH / f'ensemble_{metric_str}.npz',
            mean=metric_mean,
            sem=metric_sem,
        )

        print(f'Saved all-member, mean, and SEM arrays for {metric_str}')
        print('')


def metric_fcn_month(pred, true, time, metric_fcn, ri_t0=None):
    """
    
    """

    # Define number of month bins
    n_months = 12

    # Get month numbers from time array
    months = (time.astype('datetime64[M]').astype(int) % 12) + 1

    # Initialize list for monthly metrics
    monthly_metrics = []

    # Initialize dict for metric function keyword arguments (uncertainty)
    metric_kwargs = {}

    # Loop through months
    for i in range(n_months):

        # Get current month's time indices
        month_indices = months == (i + 1)

        if ri_t0 is not None:
            # Get uncertainty month indices
            metric_kwargs['r'] = ri_t0[month_indices]

        # Compute metric for the current month and include uncertainty kwarg if weighted metric
        # (height, width)
        month_metric = metric_fcn(
            pred[month_indices],
            true[month_indices],
            **metric_kwargs,
        )

        # Append to the list of monthly metrics
        monthly_metrics.append(month_metric)

    # Return stacked array of montly metrics along first (month) axis
    return(np.stack(monthly_metrics, axis=0)) # (month, channel, height, width)


if __name__ == '__main__':
    main()
