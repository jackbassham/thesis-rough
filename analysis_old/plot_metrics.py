import cmocean as cmo
import numpy as np
from pathlib import Path
import sys

from analysis import plot_fcns

MODEL_STRS = ['ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd']

METRIC_STRS = [
    'correlation',
    'weighted_correlation',
    'skill',
    'weighted_skill',
    'rmse',
]

ROOT = Path('/data/globus/jbassham/thesis-rough')

HEMISPHERE = 'south'
TIMESTAMP = '05222026_1652'
TIMESTAMP_REGRID = TIMESTAMP
N_MEMBERS = 10

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/rough_plots/metrics')


def metric_plot_settings(metric_str):
    """Return plotting settings for each metric."""

    if 'correlation' in metric_str:
        return dict(
            cmap=cmo.cm.balance_r,
            vmin=-1,
            vmax=1,
            cbar_label='Correlation',
        )

    if 'skill' in metric_str:
        return dict(
            cmap=cmo.cm.balance_r,
            vmin=-1,
            vmax=1,
            cbar_label='Skill',
        )

    if metric_str == 'rmse':
        return dict(
            cmap=cmo.cm.amp,
            vmin=0,
            vmax=None,
            cbar_label='RMSE',
        )

    raise ValueError(f'No plot settings defined for {metric_str}')


def main():

    # Optional command-line model selection
    try:
        model_idx = int(sys.argv[1])
        model_strs = [MODEL_STRS[model_idx]]
    except (IndexError, ValueError):
        model_strs = MODEL_STRS

    # Load lon/lat
    regrid_path = ROOT / 'regrid' / HEMISPHERE / TIMESTAMP_REGRID

    coordinates = np.load(regrid_path / 'coordinates.npz')

    lon = coordinates['lon']
    lat = coordinates['lat']

    # Make base plot directory
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    for model_str in model_strs:

        metric_path = (
            ROOT
            / 'analysis'
            / 'metrics'
            / model_str
            / HEMISPHERE
            / TIMESTAMP
        )

        model_plot_path = PLOT_PATH / model_str / HEMISPHERE / TIMESTAMP
        model_plot_path.mkdir(parents=True, exist_ok=True)

        print(f'~~~~~~~~ {model_str.upper()} ~~~~~~~~')

        for metric_str in METRIC_STRS:

            ensemble_file = metric_path / f'ensemble_{metric_str}.npz'

            if not ensemble_file.exists():
                print(f'Skipping missing file: {ensemble_file}')
                continue

            data = np.load(ensemble_file)

            metric_mean = data['mean']
            metric_sem = data['sem']

            settings = metric_plot_settings(metric_str)

            titles = ['u', 'v']

            # ------------------
            # Mean ensemble plot
            # ------------------
            mean_save_path = model_plot_path / f'ensemble_mean_{metric_str}.png'

            plot_fcns.plot_cartopy_map(
                metric_mean,
                lon,
                lat,
                hemisphere=HEMISPHERE,
                titles=titles,
                suptitle=f'{model_str.upper()} ensemble mean {metric_str}',
                data_channel_axis=0,
                n_cols=2,
                n_rows=1,
                save_path=mean_save_path,
                annotate_means=True,
                **settings,
            )

            print(f'Saved {mean_save_path}')

            # ------------------
            # SEM plot
            # ------------------
            if metric_sem is not None:
                sem_save_path = model_plot_path / f'ensemble_sem_{metric_str}.png'

                # SEM should usually be nonnegative and much smaller than metric range
                sem_settings = settings.copy()
                sem_settings['vmin'] = 0
                sem_settings['vmax'] = None
                sem_settings['cmap'] = cmo.cm.amp
                sem_settings['cbar_label'] = f'SEM {settings["cbar_label"]}'

                plot_fcns.plot_cartopy_map(
                    metric_sem,
                    lon,
                    lat,
                    hemisphere=HEMISPHERE,
                    titles=titles,
                    suptitle=f'{model_str.upper()} ensemble SEM {metric_str}',
                    data_channel_axis=0,
                    n_cols=2,
                    n_rows=1,
                    save_path=sem_save_path,
                    annotate_means=True,
                    **sem_settings,
                )

                print(f'Saved {sem_save_path}')

            print('')


if __name__ == '__main__':
    main()