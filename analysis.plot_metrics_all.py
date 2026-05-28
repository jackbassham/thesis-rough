import calendar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path

from analysis import plot_fcns, helpers


MODEL_STRS = ['ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd']


METRIC_STRS = [
    'skill',
    'weighted_skill',
    'correlation',
    'weighted_correlation',
    ]

ROOT = Path('/data/globus/jbassham/thesis-rough')

HEMISPHERE = 'south'
TIMESTAMP = '05222026_1652'
TIMESTAMP_REGRID = TIMESTAMP
N_MEMBERS = 10

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/final_plots2/global_metrics_monthly_all_models')


def main():

    # Load in nested dict of metrics 
    metrics = helpers.load_all_metrics(
        ROOT,
        MODEL_STRS,
        METRIC_STRS,
        HEMISPHERE,
        TIMESTAMP,
    )

    # Make base plot directory
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]


def plot_ensemble_all_models(
        metrics,
        model_strs,
        save_path,
        figsize=(10,10),
    ):
    
    ...

    fig, axs = plt.subplots(

    )

    metric_panels = [
        ('weighted_skill', 0, )
    ]