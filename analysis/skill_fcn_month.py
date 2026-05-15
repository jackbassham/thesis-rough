import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import _06_evaluate.metric_fcns
from analysis.plot import plot_cartopy_map


MODEL_STRS = ['cnn_pt', 'cnn_pt_wtd', 'lr_cf', 'lr_cf_wtd', 'ps']

ROOT = Path('/data/globus/jbassham/thesis-rough')

try:    
    model_idx = int(sys.argv[1]) 
except (IndexError, ValueError):
    model_idx = None

if model_idx is not None:
    MODEL_STR = MODEL_STRS[model_idx]
else:
    MODEL_STR = MODEL_STRS[0]
    
HEMISPHERE = 'south'
TIMESTAMP = '05082026_1807'

TIMESTAMP_REGRID = TIMESTAMP
TIMESTAMP_MODEL_INPUTS = TIMESTAMP
N_MEMBERS = 10

BASE_PATH = Path(
    ROOT
    / 'model-output'
    / MODEL_STR
    / HEMISPHERE
    / TIMESTAMP
)

ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')


def main():

    # Define list of metric strings
    # metric_strs = ['skill']
    metric_strs = ['skill', 'weighted_skill', 'correlation', 'weighted_correlation']
    # metric_strs = ['rmse', 'weighted_rmse']


    # plot_path_base = Path('/home/jbassham/jack/thesis-rough/plots/quick-eval/')
    plot_path_base = ANALYSIS_PATH
    plot_path = Path(plot_path_base / 'skill' / MODEL_STR / HEMISPHERE / TIMESTAMP / 'monthly')
    # Make plot path if it doesn't yet exist
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load lat, lon, and time coordinate variables
    data = np.load(ROOT / 'regrid' / HEMISPHERE / TIMESTAMP_REGRID / 'coordinates.npz')

    lat = data['lat']
    lon = data['lon']

    time = data['time_t0']

    # Create path to model inputs
    path_model_inputs = ROOT / 'model_inputs' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS

    # Load test split indices for month bins
    test_indices = np.load(path_model_inputs / 'indices_test.npz')

    # Load mask from features matrix
    mask_bad = np.load(path_model_inputs / 'targets_features.npz')['mask']
    # Squeeze out channel dimension
    mask_bad = np.squeeze(mask_bad, axis=1)

    # # Load in monthly mask
    # mask_bad = np.load(ANALYSIS_PATH / 'masks' / HEMISPHERE/ 'ci_mean_mask' / 'ci_mean_mask.npz')['mask_bad']

    # Load array of uncertainties
    # NOTE NOTE removing channel dimension here since uncertainty is same for both u and v
    # and uncertainty array is passed for preds and trues with channel dimension
    ri_t0 = np.load(path_model_inputs / 'targets_features.npz')['ri_t0']

    # Load preds and trues for each member 
    preds_list, trues_list = load_member_preds() # (member, time, channel, height, width)

    print('Finished loading preds and trues for all members)')
    print('')

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    # Compute monthly metrics for each ensemble member
    for metric_str in metric_strs:

        print(f'~~~~~~~~~~~{metric_str.upper()}~~~~~~~~~~~~~~')

        metric_fcn = getattr(_06_evaluate.metric_fcns, metric_str)

        monthly_all_members = []

        for m in range(N_MEMBERS):

            mask = mask_bad[test_indices[f'{m:02d}']] # (time, height, width)

            if 'weighted' in metric_str:
                r = ri_t0[test_indices[f'{m:02d}']] # (time, height, width)
                # Mask r as well
                r = np.where(mask[:, None, :, :], np.nan, r) # (time, channel, height, width)

            else:
                r = None

            preds = np.where(mask[:, None, :, :], np.nan, preds_list[m]) # (time, channel, height, width)
            trues = np.where(mask[:, None, :, :], np.nan, trues_list[m]) # (time, channel, height, width)

            monthly_metric = metric_fcn_month(
                preds,
                trues,
                time[test_indices[f'{m:02d}']],
                metric_fcn,
                r=r,
                # global_var_true=global_var_true,
            )  # (month, channel, height, width)

            print(f'Finished member {m} of {N_MEMBERS} for {metric_str}')

            monthly_all_members.append(monthly_metric)

        monthly_all_members = np.stack(monthly_all_members, axis=0) # (member, month, channel, height, width)

        # Save monthly metrics for all members
        np.savez(
            BASE_PATH / f'monthly_all_members_{metric_str}.npz',
            monthly_all_members=monthly_all_members,
        )

        print(f'Monthly metrics saved for all members for {metric_str}')

        monthly_mean = np.nanmean(monthly_all_members, axis=0)
        
        if N_MEMBERS > 1:
            monthly_sem  = np.nanstd(monthly_all_members, axis=0) / np.sqrt(N_MEMBERS)
        else:
            monthly_sem = None

        print(f'Monthly mean and SEM computed for {metric_str}')

        # Get glbal mean and SEM of the field for each month for each member
        global_per_member = np.nanmean(monthly_all_members, axis=(-1, -2))  # (member, month, channel)

        # Compute the global mean and SEM of the field across members for each month
        global_monthly_mean = np.nanmean(global_per_member, axis=0)
        global_monthly_sem  = np.nanstd(global_per_member, axis=0) / np.sqrt(N_MEMBERS)



        print(f'Global Monthly mean and SEM computed for {metric_str}')


        for ch, ch_name in enumerate(['u', 'v']):

            # -------- Mean plots --------
            plot_cartopy_map(
                data=monthly_mean[:, ch],   # (month, lat, lon)
                lon=lon,
                lat=lat,
                hemisphere=HEMISPHERE,
                titles=month_labels,
                suptitle=f'{metric_str.upper()} MEAN ({ch_name}): {MODEL_STR} {TIMESTAMP}',
                data_channel_axis=0,
                n_cols=4,
                n_rows=3,
                cmap=cmo.cm.balance_r, 
                vmin=-1,
                vmax=1,
                save_path=plot_path / f'{metric_str}_mean_{ch_name}.png',
            )

            if monthly_sem is not None:

                # -------- SEM plots --------
                plot_cartopy_map(
                    data=monthly_sem[:, ch],
                    lon=lon,
                    lat=lat,
                    hemisphere=HEMISPHERE,
                    titles=month_labels,
                    suptitle=f'{metric_str.upper()} SEM ({ch_name}): {MODEL_STR} {TIMESTAMP}',
                    data_channel_axis=0,
                    n_cols=4,
                    n_rows=3,
                    cmap=cmo.cm.amp,   # better for uncertainty
                    vmin=0,
                    vmax=1,
                    save_path=plot_path / f'{metric_str}_sem_{ch_name}.png',
                )

            # -------- Global MEAN/SEM plots --------
            plot_global_monthly_ensemble(
            global_mean=global_monthly_mean,
            global_sem=global_monthly_sem,
            month_labels=month_labels,
            metric_str=metric_str,
            model_str=MODEL_STR,
            save_path=plot_path / f"{metric_str}_global_monthly.png",
            )

        # ---- Optional: save arrays ----
        np.savez(
            BASE_PATH / f'{metric_str}_monthly_ensemble_mean.npz',
            mean=monthly_mean,
            sem=monthly_sem,
        )

        print(f'Finished plotting monthly mean and SEM for {metric_str}')
        print('')


def load_member_preds():
    
    # Initialize lists for metrics
    preds = []
    trues = []

    for m in range(N_MEMBERS):

        path_member = BASE_PATH / f'member_{m:02d}'

        preds_data = np.load(path_member / f'preds.npz')

        preds.append(preds_data['y_pred']) # (time, channel, height, width)
        trues.append(preds_data['y_true'])

    # Return lists of preds and trues for each member
    return preds, trues


def metric_fcn_month(pred, true, time, metric_fcn, r=None, global_var_true=None):
    """
    
    """

    # Define number of month bins
    n_months = 12

    # Get month numbers from time array
    months = (time.astype('datetime64[M]').astype(int) % 12) + 1

    # Initialize list for monthly metrics
    monthly_metrics = []

    # Loop through months
    for i in range(n_months):

        # Get current month's time indices
        month_indices = months == (i + 1)

        # Initialize dict for extra uncertainty keyword argument for weighted metrics
        metric_kwargs = {}

        if r is not None:
            # Add current month's uncertainties array to kword arguments
            metric_kwargs['r'] = r[month_indices]

        if global_var_true is not None:
            # Add current month's global_var_true array to kword arguments
            metric_kwargs['global_var_true'] = global_var_true

        # Compute metric for the current month and include uncertainty kwarg if weighted metric
        # (height, width)
        month_metric = metric_fcn(
            pred[month_indices],
            true[month_indices],
            **metric_kwargs,
        )

        # Append to the list of monthly metrics
        monthly_metrics.append(month_metric)

    # Return stacked array of montly metrics along first (month) axis
    return(np.stack(monthly_metrics, axis=0)) # (month, height, width)


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