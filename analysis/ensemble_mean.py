import numpy as np
from pathlib import Path 
from . import plot

ROOT = Path('data/globus/jbassham/thesis-rough')
MODEL_STR = ''
HEMISPHERE = ''
TIMESTAMP = ''

N_MEMBERS = 10


BASE_PATH = Path(
    ROOT
    / MODEL_STR
    / HEMISPHERE
    / TIMESTAMP
)

def main():

    # Define list of metric strings
    metric_strs = ['skill', 'wtd_skill', 'corr', 'wtd_corr']



    ...

def load_member_metrics():
    

    # Initialize lists for metrics
    u_metrics = 
    v_metrics = 

    for m in range(N_MEMBERS):

        path_member = BASE_PATH / f'member_{m:02d}'

        metric_data = np.load(path_member / f'metric_str')

        u_metrics.append(metric_data['u'])
        v_metrics.append(metric_data['v'])

    # Stack members in list into array along member axis
    u_metric_all = np.stack(u_metric_all, axis = 0)
    v_metric_all = np.stack(v_metric_all, axis = 0)

    return u_metric_all, v_metric_all


def plot_ensemble_mean_mse(metric_all, metric_str):

    mean = np.nanmean(metric_all, axis = 0)

