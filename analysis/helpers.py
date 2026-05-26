import numpy as np

def load_member_preds(n_members, base_source_path):
    
    # Initialize lists for metrics
    preds = []
    trues = []

    for m in range(n_members):

        path_member = base_source_path / f'member_{m:02d}'

        preds_data = np.load(path_member / f'preds.npz')

        preds.append(preds_data['y_pred']) # (time, channel, height, width)
        trues.append(preds_data['y_true'])

    # Return lists of preds and trues for each member
    return preds, trues


def load_member_test_split(
        source_path,
        m_key,
):
    """
    
    """

    # Get path to model inputs
    path = config.path_config.data_stage_path('model_inputs')

    # Load target feature input arrays
    input_arrays = np.load(path / 'targets_features.npz')
    

    test_indices = np.load(path / 'indices_test.npz')


    # Initialize train, val, and test split dicts
    member_splits = {
        'test': {}
    }

    # Split all arrays for all inputs included in targets and feautures (ie: 'x', 'y', 'ri_t0')
    for input_name, array in input_arrays.items():
        member_splits['test'][input_name] = array[test_indices[m_key]]

    return member_splits