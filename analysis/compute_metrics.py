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
    / 'analysis'
    / 'metrics'
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

    # Load lists of masked predictions and true values from each member
    preds_list, trues_list, ri_t0s_list = helpers.load_and_mask_member_preds(
        N_MEMBERS,
        BASE_SOURCE_PATH,
        path_model_inputs,
    )

    print(f'~~~~~All member masked preds and trues loaded~~~~~')
    
    # Compute metrics for each ensemble member
    for metric_str in metric_strs:

        print(f'~~~~~~~~~~~{metric_str.upper()}~~~~~~~~~~~~~~')

        metric_fcn = getattr(_06_evaluate.metric_fcns, metric_str)

        all_members = []

        # Initialize dict for extra uncertainty keyword argument for weighted metrics
        metric_kwargs = {}

        for m in range(N_MEMBERS):

            pred = preds_list[m]
            true = trues_list[m]

            if 'weighted' in metric_str:
                # Add current month's uncertainties array to kword arguments
                metric_kwargs['r'] = ri_t0s_list[m]

            # Compute member memtric 
            metric = metric_fcn(
                pred,
                true,
                **metric_kwargs
            )  # expected shape: (channel, height, width)

            all_members.append(metric)

            print(f'Finished member {m} of {N_MEMBERS} for {metric_str}')

        all_members = np.stack(all_members, axis=0)
        # shape: (member, channel, height, width)

        np.savez(
            BASE_DEST_PATH / f'all_members_{metric_str}.npz',
            all_members=all_members,
        )

        metric_mean = np.nanmean(all_members, axis=0)

        if N_MEMBERS > 1:
            metric_sem = np.nanstd(all_members, axis=0) / np.sqrt(N_MEMBERS)
        else:
            metric_sem = None

        np.savez(
            BASE_DEST_PATH / f'ensemble_{metric_str}.npz',
            mean=metric_mean,
            sem=metric_sem,
        )

        print(f'Saved all-member, mean, and SEM arrays for {metric_str}')
        print('')


if __name__ == '__main__':
    main()






