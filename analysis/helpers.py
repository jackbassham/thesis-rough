import numpy as np


def load_and_mask_member_preds(
        n_members,
        base_source_path,
        model_inputs_path,
        mask_norm_path,
        return_indices=False,
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

    Ui_t0 = np.load(mask_norm_path / 'global_stds.npz')['Ui_t0']

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
        mask_fixed_monthly = input_arrays['mask_fixed_monthly'][idx]
        
        # Load in uncertainty
        ri_t0 = input_arrays['ri_t0'][idx]

        # Rescale uncertainty
        ri_t0 = ri_t0 * Ui_t0

        # Combine masks
        mask = (
            mask_bad
            | mask_fixed_monthly
            | (ri_t0 >= 100)
            | (ri_t0 == 0)
        )

        # Apply mask
        y_pred = np.where(mask, np.nan, y_pred)
        y_true = np.where(mask, np.nan, y_true)
        ri_t0 = np.where(mask, np.nan, ri_t0)

        preds.append(y_pred)
        trues.append(y_true)
        ri_t0s.append(ri_t0)

    if return_indices:
        # Return lists of preds, trues, uncertainties, and indices for each member
        return preds, trues, ri_t0s, test_indices
    
    else:
        # Return lists of preds, trues, and uncertainties for each member
        return preds, trues, ri_t0s


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


from pathlib import Path
import numpy as np


def load_all_metrics(
    data_root,
    model_strs,
    metric_strs,
    hemisphere,
    timestamp,
    monthly=False
):
    """
    Returns
    -------
    metrics : dict

    Non-monthly:
        metrics[model_str][metric_str]['mean']
        metrics[model_str][metric_str]['sem']

    Monthly:
        metrics[model_str][metric_str]['mean']
        metrics[model_str][metric_str]['sem']
        metrics[model_str][metric_str]['all_members']
    """

    # Initialize empty dict for all nested metrics
    metrics = {}

    # Loop through models
    for model_str in model_strs:

        if monthly:
            base_path = (
                Path(data_root)
                / 'analysis3'
                / 'metrics_monthly'
                / model_str
                / hemisphere
                / timestamp
            )

        else:
            base_path = (
                Path(data_root)
                / 'analysis3'
                / 'metrics'
                / model_str
                / hemisphere
                / timestamp
            )

        # Initialize empty nested dict for current model's metrics
        metrics[model_str] = {}

        for metric_str in metric_strs:

            # Initialize empty nested dict for current metric
            metrics[model_str][metric_str] = {}

            # ---------- Ensemble Mean/ SEM ----------
            ensemble_path = (
                base_path
                / f'ensemble_{metric_str}.npz'
            )

            if ensemble_path.exists():

                ensemble_data = np.load(ensemble_path)

                metrics[model_str][metric_str]['mean'] = (
                    ensemble_data['mean']
                )

                metrics[model_str][metric_str]['sem'] = (
                    ensemble_data['sem']
                )

            else:
                raise FileNotFoundError(f'Ensemble path does not exist: {ensemble_path}')

            # ---------- All Members ----------
            if monthly:

                members_path = (
                    base_path
                    / f'monthly_all_members_{metric_str}.npz'
                )

                if members_path.exists():

                    members_data = np.load(members_path)

                    metrics[model_str][metric_str]['all_members'] = (
                        members_data['monthly_all_members']
                    )

                    members_path = (
                        base_path
                        / f'ensemble_{metric_str}.npz'
                    )

                    members_data = np.load(members_path)

                    metrics[model_str][metric_str]['mean'] = (
                        members_data['mean']
                    )

                    metrics[model_str][metric_str]['sem'] = (
                        members_data['sem']
                    )

                else:
                    raise FileNotFoundError(f'Ensemble path does not exist: {ensemble_path}')


    return metrics


def compute_global_monthly_metrics(metrics, n_members):
    """
    Compute global monthly mean and 2*SEM for each model/metric.

    Input:
        metrics[model_str][metric_str]['all_members']
            shape: (member, month, channel, lat, lon)
            or     (member, month, lat, lon)

    Output:
        global_metrics[model_str][metric_str]['mean']
        global_metrics[model_str][metric_str]['sem']
    """

    global_metrics = {}

    for model_str, model_metrics in metrics.items():

        global_metrics[model_str] = {}

        for metric_str, metric_data in model_metrics.items():

            print()
            print(model_str, metric_str, metric_data.keys())
            print()

            monthly_all_members = metric_data['all_members']

            # Spatial mean for each member/month/channel
            global_per_member = np.nanmean(
                monthly_all_members,
                axis=(-1, -2)
            )

            # Ensemble statistics across members
            global_monthly_mean = np.nanmean(
                global_per_member,
                axis=0
            )

            global_monthly_sem = (
                np.nanstd(global_per_member, axis=0)
                / np.sqrt(n_members)
            )

            global_metrics[model_str][metric_str] = {
                'mean': global_monthly_mean,
                'sem': global_monthly_sem,
            }

    return global_metrics