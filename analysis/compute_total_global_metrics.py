import numpy as np
from pathlib import Path

import _06_evaluate.metric_fcns
from analysis import helpers


MODEL_STRS = ['ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd']

DATA_ROOT = Path('/data/globus/jbassham/thesis-rough')
HEMISPHERE = 'south'
TIMESTAMP = '06082026_1154'
TIMESTAMP_MODEL_INPUTS = TIMESTAMP
N_MEMBERS = 10

METRIC_STRS = [
    'correlation',
    'weighted_correlation',
    'skill',
    'weighted_skill',
    'rmse',
]

ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis3')


BASE_DEST_PATH = (
    ANALYSIS_PATH
    / 'summary'
)


def main():

    BASE_DEST_PATH.mkdir(parents=True, exist_ok=True)

    path_model_inputs = DATA_ROOT / 'model_inputs' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS
    path_mask_norm = DATA_ROOT / 'mask_norm' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS


    summary_lines = []
    summary_lines.append(f'Global ensemble metric summary')
    summary_lines.append(f'Hemisphere: {HEMISPHERE}')
    summary_lines.append(f'Timestamp: {TIMESTAMP}')
    summary_lines.append(f'N members: {N_MEMBERS}')
    summary_lines.append('')

    for model_str in MODEL_STRS:

        print(f'========== {model_str} ==========')

        base_source_path = (
            DATA_ROOT
            / 'model-output'
            / model_str
            / HEMISPHERE
            / TIMESTAMP
        )

        preds_list, trues_list, ri_t0s_list = helpers.load_and_mask_member_preds(
            N_MEMBERS,
            base_source_path,
            path_model_inputs,
            path_mask_norm
        )

        summary_lines.append(f'Model: {model_str}')

        for metric_str in METRIC_STRS:

            print(f'Computing {metric_str}')

            metric_fcn = getattr(_06_evaluate.metric_fcns, metric_str)

            member_global_means = []

            for m in range(N_MEMBERS):

                pred = preds_list[m]
                true = trues_list[m]

                metric_kwargs = {}

                if 'weighted' in metric_str:
                    metric_kwargs['r'] = ri_t0s_list[m]

                metric = metric_fcn(
                    pred,
                    true,
                    **metric_kwargs,
                )

                # scalar mean over channel, lat, lon
                member_global_mean = np.nanmean(metric)
                member_global_means.append(member_global_mean)

            member_global_means = np.array(member_global_means)

            ensemble_mean = np.nanmean(member_global_means)

            if N_MEMBERS > 1:
                # ddof for sample standard deviation
                ensemble_sem = np.nanstd(member_global_means, ddof=1) / np.sqrt(N_MEMBERS)
                emsemble_2sigma = 2 * ensemble_sem
            else:
                ensemble_sem = np.nan
                ensemble_2sigma = np.nan

            summary_lines.append(
                f'  {metric_str}: mean = {ensemble_mean:.6f}, SEM = {ensemble_sem:.6f}, 2sigma = {emsemble_2sigma:.6f}'
            )

        summary_lines.append('')

    summary_path = BASE_DEST_PATH / 'global_ensemble_metric_summary.txt'

    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f'Saved summary to: {summary_path}')


if __name__ == '__main__':
    main()