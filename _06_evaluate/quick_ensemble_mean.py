import numpy as np

from _00_config.parse_args import parse_args
from .quick_eval import plot_metric


def main(cfg):

    # Instantiate argumnet parser
    args = parse_args()

    # Run model 
    run_ensemble_mean(cfg, model_name = args.model_name)


def load_member_metrics(
        metric_str,
        config,
        model_name,
    ):

    # Get number of members from config
    members = config.split_config.n_members

    u_metric_all = []
    v_metric_all = []

    for m in range(members):

        # Update runtime for the member
        config.runtime.member = m

        # Load path for the member
        path_model = config.path_config.model_path(model_name)

        # Load the metric array
        metric_arr = np.load(path_model / f'{metric_str}.npz')

        # Append to the list
        u_metric_all.append(metric_arr['u'])
        v_metric_all.append(metric_arr['v'])

    # Stack members in list into array along member axis
    u_metric_all = np.stack(u_metric_all, axis = 0)
    v_metric_all = np.stack(v_metric_all, axis = 0)

    return u_metric_all, v_metric_all


def compute_ensemble_mean_std_error(
        metric_all,
        n_members,
):
    """

    """

    mean = np.nanmean(metric_all, axis=0)
    sem = np.nanvar(metric_all, axis = 0) / np.sqrt(n_members)


def load_compute_and_plot(metric_str, config, model_name, n_members):

    u_metric_all, v_metric_all = load_member_metrics(
        metric_str, config, model_name,
    )

    # Plot ensemble means
    plot_metric(
        np.mean(u_metric_all, axis = 0),
    )

def run_ensemble_mean(config, model_name: str):

    # Get number of members
    n_members = config.split_config.n_members

    u_skill_all, v_skill_all = load_member_metrics(
        'skill', config, model_name
    )
    
    mean_u_skll, mse_u_skill = compute_ensemble_mean_std_error(
        u_skill_all, n_members
    )

    mean_u_skll, mse_u_skill = compute_ensemble_mean_std_error(
        u_skill_all, n_members
    )
