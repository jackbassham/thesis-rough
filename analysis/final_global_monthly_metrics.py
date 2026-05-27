import calendar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

from analysis import plot_fcns, helpers


# MODEL_STRS = ['ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd']
MODEL_STRS = ['ps', 'lr_cf_wtd', 'cnn_pt_wtd']
# MODEL_STRS = ['ps', 'lr_cf', 'cnn_pt']


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

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/final_plots/global_metrics_monthly_wtd_models')


def main():

    # Load in nested dict of metrics 
    metrics = helpers.load_all_metrics(
        ROOT,
        MODEL_STRS,
        METRIC_STRS,
        HEMISPHERE,
        TIMESTAMP,
        monthly=True
    )

    global_metrics = compute_global_monthly_metrics(metrics, N_MEMBERS)

    # Make base plot directory
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    plot_global_monthly_ensemble_all_models(
            global_metrics=global_metrics,
            model_strs=MODEL_STRS,
            month_labels=month_labels,
            metric_strs=['weighted_correlation', 'weighted_skill'],
            save_path=PLOT_PATH / f'global_monthly_all_models.png',
            ylabel="Skill",
    )


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


def plot_global_monthly_ensemble_all_models(
        global_metrics,
        model_strs,
        metric_strs,
        month_labels,
        save_path,
        channels=("u", "v"),
        model_colors=None,
        figsize=(12, 10),
    ):

    months = np.arange(1, 13)

    if model_colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        model_colors = {
            model_str: default_colors[i % len(default_colors)]
            for i, model_str in enumerate(model_strs)
        }

    linestyles = {
        "u": "-",
        "v": "--",
    }

    n_metrics = len(metric_strs)

    fig, axs = plt.subplots(
        n_metrics,
        1,
        figsize=figsize,
        sharex=True,
    )

    # Handle single subplot case
    if n_metrics == 1:
        axs = [axs]

    for ax, metric_str in zip(axs, metric_strs):

        for model_str in model_strs:

            global_mean = global_metrics[model_str][metric_str]["mean"]

            # NOTE using 2 sigma
            global_sem = (
                2 * global_metrics[model_str][metric_str]["sem"]
            )

            for ch, channel in enumerate(channels):

                ax.errorbar(
                    months,
                    global_mean[:, ch],
                    yerr=global_sem[:, ch],
                    color=model_colors[model_str],
                    linestyle=linestyles[channel],
                    capsize=3,
                    linewidth=2,
                )

        ax.set_ylabel(metric_str.upper())
        ax.grid(True, alpha=0.3)

    # Bottom axis only
    axs[-1].set_xticks(months)
    axs[-1].set_xticklabels(month_labels, rotation=45)
    axs[-1].set_xlabel("Month")

    # -----------------------
    # Shared legends
    # -----------------------
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D(
            [0], [0],
            color=model_colors[model_str],
            lw=2,
            label=model_str
        )
        for model_str in model_strs
    ]

    channel_handles = [
        Line2D(
            [0], [0],
            color="black",
            lw=2,
            linestyle=linestyles[channel],
            label=channel
        )
        for channel in channels
    ]

    legend1 = fig.legend(
        handles=model_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.4, 0.98),
        ncol=len(model_strs),
    )

    fig.add_artist(legend1)

    fig.legend(
        handles=channel_handles,
        title="Channel",
        loc="upper center",
        bbox_to_anchor=(0.9, 0.98),
        ncol=len(channels),
    )

    fig.suptitle(
        "Global Monthly Metrics",
        fontweight="bold",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
