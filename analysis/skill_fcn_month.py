import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import _06_evaluate.metric_fcns
from analysis.plot import plot_cartopy_map


MODEL_STRS = ['cnn_pt', 'cnn_pt_wtd', 'lr_cf', 'lr_cf_wtd', 'ps']

ROOT = Path('/data/globus/jbassham/thesis-rough')
MODEL_STR = MODEL_STRS[0]
HEMISPHERE = 'south'
TIMESTAMP = '05012026_1459'

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


def main():

    # Define list of metric strings
    metric_strs = ['skill', 'wtd_skill', 'corr', 'wtd_corr']

    plot_path_base = Path('/home/jbassham/jack/thesis-rough/plots/quick-eval/')
    plot_path = plot_path_base / MODEL_STR / HEMISPHERE / TIMESTAMP / 'monthly'
    # Make plot path if it doesn't yet exist
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load lat, lon, and time coordinate variables
    data = np.load(ROOT / 'regrid' / HEMISPHERE / TIMESTAMP_REGRID / 'coordinates.npz')

    lat = data['lat']
    lon = data['lon']

    time = data['time_t0']

    # Load test split indices for month bins
    test_indices = np.load(ROOT / 'model_inputs' / HEMISPHERE / TIMESTAMP_MODEL_INPUTS / 'indices_test.npz')

    # Load preds and trues for each member 
    preds_list, trues_list = load_member_preds() # (member, time, channel, height, width)

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    # Compute monthly metrics for each ensemble member
    for metric_str in metric_strs:

        metric_fcn = getattr(_06_evaluate.metric_fcns, metric_str)

        monthly_all_members = []

        for m in range(N_MEMBERS):

            monthly_metric = metric_fcn_month(
                preds_list[m],
                trues_list[m],
                time[test_indices[f'm:02d']],
                metric_fcn,
            )  # (month, channel, height, width)

            monthly_all_members.append(monthly_metric)

        monthly_all_members = np.stack(monthly_all_members, axis=0) # (member, month, channel, height, width)

        # Save monthly metrics for all members
        np.savez(
            BASE_PATH / f'monthly_all_members_{metric_str}.npz',
            monthly_all_members=monthly_all_members,
        )

        monthly_mean = np.nanmean(monthly_all_members, axis=0)
        monthly_std  = np.nanstd(monthly_all_members, axis=0)
        monthly_sem  = monthly_std / np.sqrt(N_MEMBERS)


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
                vmax=np.nanmax(monthly_sem[:, ch]),
                save_path=plot_path / f'{metric_str}_sem_{ch_name}.png',
            )

        # ---- Optional: save arrays ----
        np.savez(
            BASE_PATH / f'{metric_str}_monthly_ensemble_mean.npz',
            mean=monthly_mean,
            sem=monthly_sem,
        )


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


def metric_fcn_month(pred, true, time, metric_fcn, r=None):
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


if __name__ == '__main__':
    main()