import numpy as np


def load_and_mask_member_preds(
        n_members,
        base_source_path,
        model_inputs_path,
):
    """
    Load predictions/truths for each ensemble member and apply
    combined evaluation masks.
    """

    # Load target/feature arrays and member test indices
    input_arrays = np.load(model_inputs_path / 'targets_features.npz')
    test_indices = np.load(model_inputs_path / 'indices_test.npz')

    preds = []
    trues = []
    ri_t0s = []

    for m in range(n_members):

        m_key = f'{m:02d}'

        # Load predictions
        path_member = base_source_path / f'member_{m_key}'
        preds_data = np.load(path_member / 'preds.npz')

        y_pred = preds_data['y_pred']
        y_true = preds_data['y_true']

        # Get member test slice indices
        idx = test_indices[m_key]

        # Load masks and slice to member test indices
        mask_bad = input_arrays['mask_bad'][idx]
        fixed_monthly_mask = input_arrays['fixed_monthly_mask'][idx]
        
        # Load in uncertainty
        ri_t0 = input_arrays['ri_t0'][idx]

        # Combine bad point and fixed monthly masks
        mask = mask_bad | fixed_monthly_mask

        # # Expand channel dimension
        # mask = mask[:, np.newaxis, :, :]

        # Apply mask
        y_pred = np.where(mask, np.nan, y_pred)
        y_true = np.where(mask, np.nan, y_true)
        ri_t0 = np.where(mask, np.nan, ri_t0)

        preds.append(y_pred)
        trues.append(y_true)
        ri_t0s.append(ri_t0)

    # Return lists of preds and trues for each member
    return preds, trues


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

    # Load target feature input arrays
    input_arrays = np.load(source_path / 'targets_features.npz')
    

    test_indices = np.load(source_path / 'indices_test.npz')


    # Initialize train, val, and test split dicts
    member_splits = {
        'test': {}
    }

    # Split all arrays for all inputs included in targets and feautures (ie: 'x', 'y', 'ri_t0')
    for input_name, array in input_arrays.items():
        member_splits['test'][input_name] = array[test_indices[m_key]]

    return member_splits