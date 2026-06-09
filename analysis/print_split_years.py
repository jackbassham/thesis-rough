import numpy as np
from pathlib import Path

import _06_evaluate.metric_fcns
from analysis import helpers


DATA_ROOT = Path('/data/globus/jbassham/thesis-rough')
HEMISPHERE = 'south'
TIMESTAMP = '06082026_1154'
TIMESTAMP_MODEL_INPUTS = TIMESTAMP
N_MEMBERS = 10

ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')


BASE_DEST_PATH = (
    ANALYSIS_PATH
)


def main():

    BASE_DEST_PATH.mkdir(parents=True, exist_ok=True)

    path_model_inputs = DATA_ROOT / 'model_inputs' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS

    fnam_pfx = 'split_years_meta_'

    train_meta = np.load(path_model_inputs / (fnam_pfx + 'train.npz'))
    val_meta = np.load(path_model_inputs / (fnam_pfx + 'val.npz'))
    test_meta = np.load(path_model_inputs / (fnam_pfx + 'test.npz'))


    summary_lines = []
    summary_lines.append(f'Split Years')
    summary_lines.append(f'Hemisphere: {HEMISPHERE}')
    summary_lines.append(f'Timestamp: {TIMESTAMP}')
    summary_lines.append(f'N members: {N_MEMBERS}')
    summary_lines.append('')


    for m in range(N_MEMBERS):

        m_str = f'{m:02d}'

        print(f'========== member {m_str} ==========')

        summary_lines.append(f'Member: {m_str}')

        summary_lines.append(
            f'Train: {train_meta[m_str]}'
        )

        summary_lines.append(
            f'Val: {val_meta[m_str]}'
        )

        summary_lines.append(
            f'Test: {test_meta[m_str]}'
        )

        summary_lines.append('')

    summary_path = BASE_DEST_PATH / 'split_years_meta.txt'

    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f'Saved summary to: {summary_path}')


if __name__ == '__main__':
    main()