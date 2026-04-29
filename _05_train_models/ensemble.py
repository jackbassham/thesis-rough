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
    train_indices = np.load(path / 'indices_train.npz')
    val_indices = np.load(path / 'indices_val.npz')
    test_indices = np.load(path / 'indices_test.npz')

    # Get current ensemble member
    member = config.runtime.member

    # Get key from ensemble member in format ##
    m_key = f'{member:02d}'


    # Initialize train, val, and test split dicts
    member_splits = {
        'train': {},
        'val': {},
        'test': {}
    }

    # Split all arrays for all inputs included in targets and feautures (ie: 'x', 'y', 'ri_t0')
    for input_name, array in input_arrays.items():
        member_splits['train'][input_name] = array[train_indices[m_key]]
        member_splits['val'][input_name] = array[val_indices[m_key]]
        member_splits['test'][input_name] = array[test_indices[m_key]]

    return member_splits