import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import _06_evaluate.metric_fcns
from analysis.plot import plot_cartopy_map




ROOT = Path('/data/globus/jbassham/thesis-rough')


HEMISPHERE = 'south'

TIMESTAMP_MASK_NORM = '05062026_1852'
TIMESTAMP_REGRID = '05062026_1852'

ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')

SAVE_PATH = ANALYSIS_PATH / 'rescale_check' / HEMISPHERE
SAVE_PATH.mkdir(parents=True, exist_ok=True)



def main():

    # Define list of metric strings
    metric_strs = ['skill']
    # metric_strs = ['rmse', 'weighted_rmse', 'mae', 'mean_misfit']


    # Load stats for rescaling predictions/ trues
    path_stats = ROOT / 'mask_norm' / HEMISPHERE / TIMESTAMP_MASK_NORM

    # Standard deviation of speed for rescaling
    Ui_t0 = np.load(path_stats / 'global_stds.npz')['Ui_t0']

    print('~~~~~~~~~~~~~Standard Deviation of Speed~~~~~~~~~~~~~~~~~~')
    print(f'Ui_t0: {Ui_t0}')

    # Load gridwise means for rescaling
    gridwise_means = dict(np.load(path_stats / 'gridwise_means.npz'))

    # Get list of variable names from np file keys
    var_names = list(gridwise_means.keys())

    # Stack the arrays into a data array shaped (variable, ny, nx)
    mean_data = np.stack([gridwise_means[name] for name in var_names], axis=0)

    # Load lat, lon, and time coordinate variables
    data = np.load(ROOT / 'regrid' / HEMISPHERE / TIMESTAMP_REGRID / 'coordinates.npz')

    lat = data['lat']
    lon = data['lon']

    print('~~~~~~~~~~~~~Plotting Gridwise Means~~~~~~~~~~~~~~~~~~')

    plot_cartopy_map(
        data=mean_data, 
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=var_names,
        suptitle=f'Gridwise Means for Scaling',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal, 
        cbar_label='cm_s',
        vmin=-10,
        vmax=10,
        save_path=SAVE_PATH / 'gridwise_means.png',
    )

    
if __name__ == '__main__':
    main()