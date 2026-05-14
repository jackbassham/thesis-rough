import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import _06_evaluate.metric_fcns




ROOT = Path('/data/globus/jbassham/thesis-rough')


HEMISPHERE = 'south'

TIMESTAMP_MASK_NORM = '05062026_1852'


def main():

    # Define list of metric strings
    metric_strs = ['skill']
    # metric_strs = ['rmse', 'weighted_rmse', 'mae', 'mean_misfit']


    # Load stats for rescaling predictions/ trues
    path_stats = ROOT / 'mask_norm' / HEMISPHERE / TIMESTAMP_MASK_NORM

    # Standard deviation of speed for rescaling
    Ui_t0 = np.load(path_stats / 'global_stds.npz')['Ui_t0']

    print('~~~~~~~~~~~~~~~~~STATS~~~~~~~~~~~~~~~~~~')
    print(f'Ui_t0: {Ui_t0}')


if __name__ == '__main__':
    main()