import calendar
import cmocean as cmo
import matplotlib.pyplot as plt
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

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    for model_str in model_strs:

        metric_path = (
            ROOT
            / 'analysis'
            / 'metrics_monthly'
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

            # Get monthly metrics for all members
            all_members_file = metric_path / f'monthly_all_members_{metric_str}.npz' 

            if not ensemble_file.exists():
                print(f'Skipping missing file: {ensemble_file}')
                continue
            
            monthly_all_members = np.load(all_members_file)['monthly_all_members']

            # Compute global mean and SEM of the field for each month for each member
            global_per_member = np.nanmean(monthly_all_members, axis=(-1, -2))  # (member, month, channel)

            # Compute the global mean and SEM of the field across members for each month
            global_monthly_mean = np.nanmean(global_per_member, axis=0)
            global_monthly_sem  = np.nanstd(global_per_member, axis=0) / np.sqrt(N_MEMBERS)

            settings = metric_plot_settings(metric_str)

            # ------------------
            # Mean ensemble plot
            # ------------------

            for ch, ch_name in enumerate(['u', 'v']):

                mean_save_path = model_plot_path / f'ensemble_sem_{metric_str}_{ch_name}.png'

                # -------- Mean plots --------
                plot_fcns.plot_cartopy_map(
                    data=metric_mean[:, ch],   # (month, lat, lon)
                    lon=lon,
                    lat=lat,
                    hemisphere=HEMISPHERE,
                    titles=month_labels,
                    suptitle=f'{metric_str.upper()} MEAN ({ch_name}): {model_str} {TIMESTAMP}',
                    data_channel_axis=0,
                    n_cols=4,
                    n_rows=3,
                    save_path=mean_save_path,
                    **settings
                )

                print(f'Saved {mean_save_path}')

                # ------------------
                # SEM plot
                # ------------------
                if metric_sem is not None:

                    mean_sem_path = model_plot_path / f'ensemble_sem_{metric_str}_{ch_name}.png'

                    # SEM should usually be nonnegative and much smaller than metric range
                    sem_settings = settings.copy()
                    sem_settings['vmin'] = 0
                    sem_settings['vmax'] = None
                    sem_settings['cmap'] = cmo.cm.amp
                    sem_settings['cbar_label'] = f'SEM {settings["cbar_label"]}'

                    # -------- Mean plots --------
                    plot_fcns.plot_cartopy_map(
                        data=metric_sem[:, ch],   # (month, lat, lon)
                        lon=lon,
                        lat=lat,
                        hemisphere=HEMISPHERE,
                        titles=month_labels,
                        suptitle=f'{metric_str.upper()} SEM ({ch_name}): {model_str} {TIMESTAMP}',
                        data_channel_axis=0,
                        n_cols=4,
                        n_rows=3,
                        save_path=model_plot_path / f'ensemble_sem_{metric_str}_{ch_name}.png',
                        **settings
                    )

                    print(f'Saved {model_plot_path / f'ensemble_sem_{metric_str}_{ch_name}.png'}')

                print('')

            # -------- Global MEAN/SEM plots --------
            plot_global_monthly_ensemble(
            global_mean=global_monthly_mean,
            global_sem=global_monthly_sem,
            month_labels=month_labels,
            metric_str=metric_str,
            model_str=MODEL_STR,
            save_path=plot_path / f"{metric_str}_global_monthly.png",
            )


def plot_global_monthly_ensemble(
        global_mean,
        global_sem,
        month_labels,
        metric_str,
        model_str,
        save_path,
        ylabel=None,
        title_prefix="Global monthly",
        channels=("u", "v"),
        colors=("tab:blue", "tab:orange"),
        figsize=(10, 5),
    ):
    """
    Plot global monthly ensemble means with SEM error bars.

    Parameters
    ----------
    global_mean : array, shape (month, channel)
    global_sem   : array, shape (month, channel)
    month_labels : list[str], length 12
    metric_str   : str
    model_str    : str
    save_path    : Path
    ylabel       : str or None
    channels     : tuple[str]
    colors       : tuple[str]
    """

    months = np.arange(1, global_mean.shape[0] + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for ch, (label, color) in enumerate(zip(channels, colors)):
        ax.errorbar(
            months,
            global_mean[:, ch],
            yerr=global_sem[:, ch],
            label=label,
            color=color,
            marker="o",
            capsize=3,
            linewidth=2,
        )

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, rotation=45)

    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel if ylabel is not None else metric_str.upper())

    ax.set_title(f"{title_prefix} {metric_str.upper()}: ({model_str} {TIMESTAMP})", fontweight="bold")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



if __name__ == '__main__':
    main()