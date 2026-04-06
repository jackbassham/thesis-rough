import helpers
import numpy as np
import numpy.typing as npt
from pathlib import Paths
from . import split_generators

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

    # Load model inputs deestination path
    path_model_inputs = cfg.path_config.data_stage_path('model_inputs')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_model_inputs)

    # Load in masked/ normalized input parameters
    inputs = np.load(path_mask_norm / 'masked_normalized.npz')

    # Load in mask
    mask_bad = np.load(path_mask_norm / 'masks.npz')['mask_bad']

    # Convert boolean values to 0, 1 floats
    mask_bad = mask_bad.astype(np.float32)

    # Add mask to inputs
    inputs['mask'] = mask_bad

    # Fill target, feature, and separate unertainty arrays from inputs
    arrays = make_target_feature_arrays(inputs)

    # Load regrid data source path for coordinates
    path_coordinates = cfg.path_config.data_stage_path('regrid')

    # Load in present-day time variable from coordinates
    time_t0 = np.load(path_coordinates / 'coordinates.npz')['time_t0']

    # Get split indices from time array
    indices = split_generators.chronological_indices(time_t0)

    # Split arrays based on indices
    splits = split_arrays(arrays)

    # Create the destination directory if it doesn't already exist
    os.makedirs(PATH_MODEL_INPUTS, exist_ok = True)

    # Save splits
    np.savez(
        os.path.join(PATH_MODEL_INPUTS, 'train.npz'),
        x_train = x_train,
        y_train = y_train,
        r_train = r_train,
        nan_mask_train = nan_mask_train,
    )

    np.savez(
        os.path.join(PATH_MODEL_INPUTS, 'val.npz'),
        x_val = x_val,
        y_val = y_val,
        r_val = r_val,
        nan_mask_val = nan_mask_val,
    )

    np.savez(
        os.path.join(PATH_MODEL_INPUTS, 'test.npz'),
        x_test = x_test,
        y_test = y_test,
        r_test = r_test,
        nan_mask_test = nan_mask_test,
    )

    print(f'Splits saved at {PATH_MODEL_INPUTS}')

    # Save split indices

    np.savez_compressed(
    os.path.join(PATH_MODEL_INPUTS, f"split_indices.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    
    print(f"Split indices saved at {PATH_MODEL_INPUTS}")

def make_target_feature_arrays(inputs: dict[str, npt.NDArray]
                               ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    
    """

    # Define target variables
    targets = [
        'ui_t0',
        'vi_t0',
    ]

    # Define feature variables
    features = [
        'ua_t0',
        'va_t0',
        'ci_t1',
        'mask'
    ]

    # Uncertainty lives in separate array (for now)
    uncertainty = [
        'ri_t0'
    ]

    # Get input dimensions from first input
    nt, nlat, nlon = next(iter(inputs.values())).shape

    # Initialize target, feature arrays
    y = np.zeros((nt, len(targets), nlat, nlon))
    x = np.zeros((nt, len(features), nlat, nlon))

    # Fill target array
    for i, name in enumerate(targets):
        y[:, i] = inputs[name]

    # Fill feature array
    for i, name in enumerate(features):
        x[:, i] = inputs[name]

    # Add channel dimension on uncertainty to match targets, features
    ri_t0 = inputs['ri_t0'][:, np.newaxis, :, :]

    return {
        'y': y,
        'x': x,
        'ri_t0': ri_t0
    }


def split_arrays(arrays: dict[str, npt.NDArray], indices
                 ) -> dict[str, dict[str, npt.NDArray]]:
    """
    
    """

    # Initialize train, val, and test split dics
    splits = {
        'train': {},
        'test': {},
        'val': {},
    }

    for name, array in arrays.items():
        splits['train'][name] = array[indices.train],
        splits['val'][name] = array[indices.val],
        splits['test'][name] = array[indices.test]

    return splits





if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)