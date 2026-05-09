import helpers
import numpy as np
from pathlib import Path
from . import (
    split_generators,
    utils,
)

# TODO make a split generator to split years based on data config
# to a ruffled split with 2 test, 2 val, and the rest train
# (later on, can make it work with smaller data sets, for one year etc?)

# TODO option to randomly shuffle split (2 consecutive test, 2 consecutive val, and the rest
# train) (Hoffman) to move to ensemble

# TODO ensemble to run the split generator 10 times 

# TODO experiment with complete random shuffle of years

# TODO Experiment optionally include buffer year for temporal edge leakage?

# TODO One input data file with nans, bad_mask?, and land_ocean_mask. Fill nans with 0 later (and convert
# numpy to pytorch or vice versa???), for CNN training. 
# NOTE Convert numpy to PyTorch in CNN training

# TODO Experiment with including mask in the input or a masked loss for land_ocean_mask
# (incorperate in LR and PS as well?)
# NOTE Move to:
# 1. Keep replacing NaN's in CNN with 0
# 2. Include the mask (bad_mask) as an input (1 = Valid, 0 = Missing)
# 3. Use the mask in the loss function as well
# NOTE read up on: 1. Bayesian regression, 2. heteroscedastic neural networks
# NOTE question: issues with ice edge when training, could a mask improve skill on ice edge? Especially
# when dealing with monthly skills?

# TODO memmap or torch Dataset for memmory efficiency

def main(cfg):

    # Load masked/ normalized data source path
    path_mask_norm = cfg.path_config.data_stage_path('mask_norm')

    # NOTE the dict might break here when moving to memmap, or using pickle=True
    # Load in masked/ normalized input parameters as dict
    inputs = dict(np.load(path_mask_norm / 'masked_normalized.npz'))

    # Load in mask
    mask_bad = np.load(path_mask_norm / 'masks.npz')['mask_bad']

    # Add mask to inputs
    inputs['mask'] = mask_bad

    # Fill target, feature, and separate unertainty arrays from inputs
    targets_features = utils.make_target_feature_arrays(inputs)

    # Load model inputs destination path
    path_model_inputs = cfg.path_config.data_stage_path('model_inputs')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_model_inputs)

    # Save unsplit target and feature arrays
    helpers.save_arrays(
        path_model_inputs / 'targets_features.npz',
        targets_features
        )

    # Load regrid data source path for coordinates
    path_coordinates = cfg.path_config.data_stage_path('regrid')

    # Load in present-day time variable from coordinates
    time_t0 = np.load(path_coordinates / 'coordinates.npz')['time_t0']

    # Get split indices from time array
    split_indices, split_years_meta = split_generators.k_shuffled_year_indices(
        time_t0, n_members=cfg.split_config.n_members
        )

    # Save split indices and split years meta data
    split_generators.save_member_split_indices(
        path_model_inputs,
        split_indices,
        split_years_meta
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)