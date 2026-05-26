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

TIMESTAMP = '05082026_1807'

TIMESTAMP_REGRID = '05062026_1852'
TIMESTAMP_MASK_NORM = '05062026_1852'
TIMESTAMP_MODEL_INPUTS = '05082026_1807'

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

    preds_list, trues_list = helpers.load_member_preds(
        N_MEMBERS,
        BASE_SOURCE_PATH,
    )

    # Load path to model inputs
    path_model_inputs = Path(DATA_ROOT / 'model_inputs' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS)

    # Load lists of masked predictions and true values from each member
    preds_list, trues_list = helpers.load_and_mask_member_preds(
        N_MEMBERS,
        BASE_SOURCE_PATH,
        path_model_inputs,
    )
    
    # Compute metrics for each ensemble member
    for metric_str in metric_strs:

        print(f'~~~~~~~~~~~{metric_str.upper()}~~~~~~~~~~~~~~')

        metric_fcn = getattr(_06_evaluate.metric_fcns, metric_str)

        for m in range(N_MEMBERS):


            print(f'Finished member {m} of {N_MEMBERS} for {metric_str}')











