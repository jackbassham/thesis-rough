import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path

import _06_evaluate.metric_fcns as metric_fcns
from analysis.plot import plot_cartopy_map, plot_contour_cartopy_map, plot_discrete_cartopy_map
import helpers

DATA_ROOT = Path('/data/globus/jbassham/thesis-rough')
ANALYSIS_ROOT = Path('/home/jbassham/jack/thesis-rough/analysis')


MODEL_STR = 'cnn_pt'
HEMISPHERE = 'south'
TIMESTAMP_REGRID = '05062026_1852'
TIMESTAMP_MASK_NORM = TIMESTAMP_REGRID

TIMESTAMP_MODEL_OUTPUTS = '05082026_1807'
TIMESTAMP_MODEL_INPUTS = '05082026_1807'

N_MEMBERS = 10

MODEL_OUTPUT_PATH = Path(
    DATA_ROOT
    / 'model-output'
    / MODEL_STR
    / HEMISPHERE
    / TIMESTAMP_MODEL_OUTPUTS
)

MODEL_INPUT_PATH = Path(
    DATA_ROOT
    / 'model_inputs'
    / HEMISPHERE
    / TIMESTAMP_MODEL_INPUTS
)

REGRID_PATH = Path(
    DATA_ROOT
    / 'regrid'
    / HEMISPHERE
    / TIMESTAMP_REGRID
)


ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')
SAVE_PLOT_PATH = ANALYSIS_PATH / 'monthly_var_true' / HEMISPHERE
SAVE_PLOT_PATH.mkdir(parents=True, exist_ok=True)


def main():

    # Load in coordinates
    coord_data = np.load(REGRID_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    # Load test split indices for month bins
    test_indices = np.load(MODEL_INPUT_PATH / 'indices_test.npz')

    preds_list, trues_list = load_member_preds()

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    for m in range(N_MEMBERS):

        monthly_u_var = monthly_stats(
            data=trues_list[m][:,0,:,:],
            time=time_t0[test_indices[f'{m:02d}']],
            stat_fcn=np.nanvar,
            stat_fcn_kwargs={'axis': 0}
        )

        u_var_max = np.nanmax(monthly_u_var)
        u_var_min = np.nanmin(monthly_u_var)

        monthly_v_var = monthly_stats(
            data=trues_list[m][:,1,:,:],
            time=time_t0[test_indices[f'{m:02d}']],
            stat_fcn=np.nanvar,
            stat_fcn_kwargs={'axis': 0}
        )

        print(f'~~~~~~~~~~~~~~Member: {m:02d} Stats Computed~~~~~~~~~~~~~~')

        v_var_max = np.nanmax(monthly_v_var)
        v_var_min = np.nanmin(monthly_v_var)

        plot_discrete_cartopy_map(
            data=monthly_u_var,
            lon=lon,
            lat=lat,
            hemisphere=HEMISPHERE,
            titles=month_labels,
            suptitle=f'Var(u_true_norm); Member {m:02d}',
            data_channel_axis=0,
            n_cols=4,
            n_rows=3,
            cmap=cmo.cm.thermal,
            boundaries=np.linspace(0,1,num=10),
            cbar_label='variance',
            save_path=Path(SAVE_PLOT_PATH / f'var_u_true_norm_member{m:02d}.png'),
        )

        plot_discrete_cartopy_map(
            data=monthly_v_var,
            lon=lon,
            lat=lat,
            hemisphere=HEMISPHERE,
            titles=month_labels,
            suptitle=f'Var(v_true_norm); Member {m:02d}',
            data_channel_axis=0,
            n_cols=4,
            n_rows=3,
            cmap=cmo.cm.thermal,
            boundaries=np.linspace(0,1,num=10),
            cbar_label='variance',
            save_path=Path(SAVE_PLOT_PATH / f'var_v_true_norm_member{m:02d}.png'),
        )

        print(f'~~~~~~~~~~~~~~Member: {m:02d} Plots Saved~~~~~~~~~~~~~~')
        
def monthly_stats(data, time, stat_fcn, stat_fcn_kwargs=None):
    """
    
    """

    # Define number of month bins
    n_months = 12

    # Get month numbers from time array
    months = (time.astype('datetime64[M]').astype(int) % 12) + 1

    # Initialize list for monthly metrics
    monthly_stats = []

    # Loop through months
    for i in range(n_months):

        # Get current month's time indices
        month_indices = months == (i + 1)


        # Compute stat for that month, including any keyword arguments
        if stat_fcn_kwargs is not None:
            stat = stat_fcn(data[month_indices], **stat_fcn_kwargs)
        else:
            stat = stat_fcn(data[month_indices])

        # Append to the list of monthly metrics
        monthly_stats.append(stat)

    # Return stacked array of montly metrics along first (month) axis
    return(np.stack(monthly_stats, axis=0)) # (month, height, width)


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


def monthly_metric_all_members(test_indices, preds_list, trues_list, time_t0, metric_fcn, metric_fcn_kwargs=None):

    monthly_all_members = []

    for m in range(N_MEMBERS):

        monthly_metric = metric_fcn_month(
            preds_list[m],
            trues_list[m],
            time_t0[test_indices[f'{m:02d}']],
            metric_fcn,
            metric_fcn_kwargs=metric_fcn_kwargs
        )  # (month, channel, height, width)

        monthly_all_members.append(monthly_metric)

    return(np.stack(monthly_all_members, axis=0)) # (member, month, channel, height, width)


def load_member_preds():
    
    # Initialize lists for metrics
    preds = []
    trues = []

    for m in range(N_MEMBERS):

        path_member = MODEL_OUTPUT_PATH / f'member_{m:02d}'

        preds_data = np.load(path_member / f'preds.npz')

        preds.append(preds_data['y_pred']) # (time, channel, height, width)
        trues.append(preds_data['y_true'])

    # Return lists of preds and trues for each member
    return preds, trues


if __name__ == '__main__':
    main()