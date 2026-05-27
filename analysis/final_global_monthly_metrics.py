import calendar
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from analysis import plot_fcns, helpers


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

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/final_plots/global_metrics_monthly')


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
        metric_str="skill",
        save_path=PLOT_PATH / "global_monthly_skill_all_models.png",
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
        month_labels,
        metric_str,
        save_path,
        ylabel=None,
        title_prefix="Global monthly",
        channels=("u", "v"),
        model_colors=None,
        figsize=(10, 5),
    ):
    """
    Plot global monthly ensemble means with SEM error bars for all models.

    Expected input structure
    ------------------------
    global_metrics[model_str][metric_str]['mean']  # (month, channel)
    global_metrics[model_str][metric_str]['sem']   # (month, channel)
    """

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

    fig, ax = plt.subplots(figsize=figsize)

    for model_str in model_strs:

        global_mean = global_metrics[model_str][metric_str]["mean"]
        global_sem = global_metrics[model_str][metric_str]["sem"]

        for ch, channel in enumerate(channels):

            ax.errorbar(
                months,
                global_mean[:, ch],
                yerr=global_sem[:, ch],
                label=f"{model_str} {channel}",
                color=model_colors[model_str],
                linestyle=linestyles[channel],
                marker="o",
                capsize=3,
                linewidth=2,
            )

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, rotation=45)

    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel if ylabel is not None else metric_str.upper())

    ax.set_title(
        f"{title_prefix} {metric_str.upper()}: all models",
        fontweight="bold"
    )

    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
