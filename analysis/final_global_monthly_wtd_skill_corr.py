import calendar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path

from analysis import plot_fcns, helpers


# MODEL_STRS = ['ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd']
MODEL_STRS = ['ps', 'lr_cf_wtd', 'cnn_pt_wtd']
# MODEL_STRS = ['ps', 'lr_cf', 'cnn_pt']


METRIC_STRS = [
    'weighted_correlation',
    'weighted_skill',
]

ROOT = Path('/data/globus/jbassham/thesis-rough')

HEMISPHERE = 'south'
TIMESTAMP = '05222026_1652'
TIMESTAMP_REGRID = TIMESTAMP
N_MEMBERS = 10

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/final_plots2/global_metrics_monthly_wtd_models')


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

    global_metrics = helpers.compute_global_monthly_metrics(metrics, N_MEMBERS)

    # Make base plot directory
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]


    # NOTE change png to pdf!

    plot_global_monthly_ensemble_all_models(
        global_metrics,
        MODEL_STRS,
        month_labels,
        PLOT_PATH / 'wtd_corr_skill.png',
        figsize=(7, 7),
    )


def plot_global_monthly_ensemble_all_models(
        global_metrics,
        model_strs,
        month_labels,
        save_path,
        figsize=(10, 7),
    ):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    months = np.arange(1, 13)

    # --------------------------------------------------
    # Colorblind-safe colors
    # --------------------------------------------------

    model_colors = {
        "ps": '#000000',         
        "lr_cf_wtd": "#56B4E9",  
        "cnn_pt_wtd": "#E69F00",
    }

    # --------------------------------------------------
    # Figure setup
    # --------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
    )

    # ==================================================
    # TOP PANEL
    # ==================================================

    metric_str = "weighted_skill"

    for model_str in model_strs:

        mean = global_metrics[model_str][metric_str]["mean"]
        sem  = global_metrics[model_str][metric_str]["sem"]

        # u component
        ax1.errorbar(
            months,
            mean[:, 0],
            yerr=2 * sem[:, 0],
            color=model_colors[model_str],
            linestyle="-",
            linewidth=2.5,
            capsize=3,
            clip_on=False,
        )

        # v component
        ax1.errorbar(
            months,
            mean[:, 1],
            yerr=2 * sem[:, 1],
            color=model_colors[model_str],
            linestyle=":",
            linewidth=2.5,
            capsize=3,
            clip_on=False,
        )

    ax1.set_ylabel(r"$Skill$", fontsize=12, fontweight='bold')
    # ax1.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax1.set_title('Weighted Skill', fontsize=12, fontweight='bold')

    # ax1.axhline(
    #     0,
    #     color="black",
    #     linewidth=1,
    #     alpha=0.5,
    # )

    ax1.grid(
        True,
        alpha=0.25,
        linewidth=0.5,
    )

    # ==================================================
    # BOTTOM PANEL
    # ==================================================

    metric_str = "weighted_correlation"

    for model_str in model_strs:

        mean = global_metrics[model_str][metric_str]["mean"]
        sem  = global_metrics[model_str][metric_str]["sem"]

        # u component
        ax2.errorbar(
            months,
            mean[:, 0],
            yerr=2 * sem[:, 0],
            color=model_colors[model_str],
            linestyle="-",
            linewidth=2.5,
            capsize=3,
            clip_on=False,
        )

        # v component
        ax2.errorbar(
            months,
            mean[:, 1],
            yerr=2 * sem[:, 1],
            color=model_colors[model_str],
            linestyle=":",
            linewidth=2.5,
            capsize=3,
            clip_on=False,
        )

    ax2.set_ylabel(r'$\rho$', fontsize=12, fontweight='bold')
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_title('Weighted Correlation', fontsize=12, fontweight='bold')


    # ax2.axhline(
    #     0,
    #     color="black",
    #     linewidth=1,
    #     alpha=0.5,
    # )

    ax2.grid(
        True,
        alpha=0.25,
        linewidth=0.5,
    )

    # ==================================================
    # X AXIS
    # ==================================================

    ax2.set_xticks(months)
    ax2.set_xticklabels(
        month_labels,
        rotation=45,
        ha="right",
        fontweight='bold'
    )
    ax2.set_xlim(1, 12)

    # ax2.set_xlabel("Month", fontsize=12)

    # ==================================================
    # COMBINED LEGEND
    # ==================================================

    def errorbar_proxy(color, label):
        return ax1.errorbar(
            [np.nan],
            [np.nan],
            yerr=[0.1],
            color=color,
            linestyle="-",
            linewidth=2.5,
            capsize=3,
            label=label,
        )

    legend_handles = [
        Line2D([], [], linestyle="none", label="Models"),

        errorbar_proxy(model_colors["cnn_pt_wtd"], "WCNN"),
        errorbar_proxy(model_colors["lr_cf_wtd"], "WLR"),
        errorbar_proxy(model_colors["ps"], "PS"),

        Line2D([], [], linestyle="none", label="Predictions"),

        Line2D([0], [0], color="black", linestyle="-", lw=2, label=r"$u_{i,t}$"),
        Line2D([0], [0], color="black", linestyle=":", lw=2, label=r"$v_{i,t}$"),
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        fontsize=10,
    )

    # Bold legend section headers
    legend_texts = legend.get_texts()

    legend_texts[0].set_fontweight("bold")  # Models
    legend_texts[4].set_fontweight("bold")  # Predictions

    # ==================================================
    # FINAL FORMATTING
    # ==================================================

    # fig.suptitle(
    #     "Global Monthly Sea Ice Motion Metrics",
    #     fontsize=14,
    #     fontweight="bold",
    # )

    plt.tight_layout(
        rect=[0, 0, 0.86, 0.96]
    )

    # vector graphics for publications
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == '__main__':
    main()
