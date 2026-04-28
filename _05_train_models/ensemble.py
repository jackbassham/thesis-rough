import numpy as np


def run_ensemble(config, train_fcn):
    """
    
    """

    # Loop through the numer of ensemble members defined in split configuration
    for m in range(config.split_config.n_members):

        print(f'Running ensemble member {m:02d}')

        # Update runtime state so data loaders and path builders use current member
        config.runtime.member = m

        # Call the model traing function
        train_fcn(config)


def load_member_splits(config):
    """
    
    """

    # Get path to model inputs
    path = config.path_config.data_stage_path('model_inputs')

    # Load target feature input arrays
    input_arrays = np.load(path / 'targets_features.npz')
    
    # Load array of split indices for all members
    indices = np.load(path / 'split_indices.npz')

    # Get current ensemble member
    m = config.runtime.member

    # Initialize train, val, and test split dicts
    member_splits = {
        'train': {},
        'val': {},
        'test': {}
    }

    # Split all arrays for all inputs included in targets and feautures (ie: 'x', 'y', 'ri_t0')
    for input_name, array in input_arrays.items():
        member_splits['train'][input_name] = array[indices['train'][m]]
        member_splits['val'][input_name] = array[indices['val'][m]]
        member_splits['test'][input_name] = array[indices['test'][m]]

    return member_splits