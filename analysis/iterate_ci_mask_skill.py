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
    / MODEL_STR
    / HEMISPHERE
    / TIMESTAMP_MODEL_OUTPUTS
)

REGRID_PATH = Path(
    DATA_ROOT
    / 'regrid'
    / HEMISPHERE
    / TIMESTAMP_REGRID
)


ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')
SAVE_PLOT_PATH = ANALYSIS_PATH / 'mask_skill_iterate' / MODEL_STR / HEMISPHERE
SAVE_PLOT_PATH.mkdir(parents=True, exist_ok=True)


def main():

    # Load and shift ice concentration data
    ci = helpers.load_ice_conc(REGRID_PATH, 'ice_conc_regrid_nsidc0051_v2.npz')
    ci_t0 = present_day(ci)

    # Load and shift ice velocity data
    ui, vi, _ = helpers.load_ice_vel(REGRID_PATH, 'ice_vel_regrid_nsidc0016_v4.npz')
    ui_t0, vi_t0 = present_day(ui), present_day(vi)

    # Mask using steps taken in mask_normalize
    ci_t0, _ = pre_mask_raw_ci(ci_t0)

    # Load in coordinates
    coord_data = np.load(REGRID_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    # Load test split indices for month bins
    test_indices = np.load(MODEL_INPUT_PATH / 'indices_test.npz')

    preds_list, trues_list = load_member_preds()

    print('~~~~~~~~~~~Data loaded~~~~~~~~~~~')

    perc_thresh_range = np.arange(20, 70 + 5, 5)
    ci_thresh_range = np.arange(15, 40 + 5, 5)

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    # Loop over thresholds
    for perc_thresh in perc_thresh_range:
        for ci_thresh in ci_thresh_range:

            # Compute mask for the theshold combo
            mask_bad = create_monthly_ci_masks(
                ci_t0=ci_t0,
                ui_t0=ui_t0,
                vi_t0=vi_t0,
                time_t0=time_t0,
                ci_thresh=ci_thresh,
                perc_thresh=perc_thresh
            )

            # Compute the monthly skill for the threshold combo
            monthly_all_members = compute_monthly_skill(
                test_indics=test_indices,
                mask_bad=mask_bad,
                preds_list=preds_list,
                trues_list=trues_list,
                time_t0=time_t0,
                month_labels=month_labels,
                perc_thresh=perc_thresh,
                ci_thresh=ci_thresh,
            )

            monthly_mean = np.nanmean(monthly_all_members, axis=0)
            monthly_sem  = np.nanstd(monthly_all_members, axis=0) / np.sqrt(N_MEMBERS)

            # Get glbal mean and SEM of the field for each month for each member
            global_per_member = np.nanmean(monthly_all_members, axis=(-1, -2))  # (member, month, channel)

            # Compute the global mean and SEM of the field across members for each month
            global_monthly_mean = np.nanmean(global_per_member, axis=0)
            global_monthly_sem  = np.nanstd(global_per_member, axis=0) / np.sqrt(N_MEMBERS)

            # Plot
            plot(
                monthly_mean=monthly_mean,
                monthly_sem=monthly_sem,
                global_monthly_mean=global_monthly_mean,
                global_monthly_sem=global_monthly_sem,
                lat=lat,
                lon=lon,
                month_labels=month_labels,
                perc_thresh=perc_thresh,
                ci_thresh=ci_thresh,
            )

        print(f'~~~~~~perc_thresh: {perc_thresh}, ci_thresh: {ci_thresh} complete~~~~~~')


def create_monthly_ci_masks(
        ci_t0, ui_t0, vi_t0, time_t0,
        ci_thresh,
        perc_thresh,
        ):


        # Create monthly (pooled accross years) ice concentration mask based on percent days ice free
        full_monthly_mask, _ = monthly_mask(ci_t0, time_t0, perc_thresh=perc_thresh, ci_thresh=ci_thresh)

        # Use monthly mask and additional criteria to mask create total mask of bad points
        mask_bad = mask_ci(ci_t0, ui_t0, vi_t0, full_monthly_mask, ci_thresh=ci_thresh)

        return mask_bad


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


def compute_monthly_skill(test_indices, mask_bad, preds_list, trues_list, time_t0, month_labels, perc_thresh, ci_thresh):

    monthly_all_members = []

    for m in range(N_MEMBERS):

        mask = mask_bad[test_indices[f'{m:02d}']] # (time, height, width)

        preds = np.where(mask[:, None, :, :], np.nan, preds_list[m]) # (time, channel, height, width)
        trues = np.where(mask[:, None, :, :], np.nan, trues_list[m]) # (time, channel, height, width)

        monthly_metric = metric_fcn_month(
            preds,
            trues,
            time_t0[test_indices[f'{m:02d}']],
            metric_fcns.skill,
        )  # (month, channel, height, width)

        monthly_all_members.append(monthly_metric)

        return(np.stack(monthly_all_members, axis=0)) # (member, month, channel, height, width)
    

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

    ax.set_title(f"{title_prefix} SKILL: ({model_str})", fontweight="bold")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[:-1,:,:]    


def perc_days_ice_free(ci, threshold=0.15):

    # Determine valid, non-nan ice conentration grid points
    valid = ~np.isnan(ci)
    
    # Sum total number of valid days at each gridpoint
    n_total = np.sum(valid, axis=0)

    # Sum number of valid ice free days at each gridpoint
    n_ice_free = np.sum((ci <= threshold) & valid, axis=0)

    # Initialize array of nans for percent ice free
    perc_days_ice_free = np.full_like(n_total, np.nan, dtype=np.float32)

    # Divide where valid points exist, otherwise leave nan
    np.divide(
        n_ice_free * 100,
        n_total,
        out=perc_days_ice_free,
        where=n_total != 0
    )
            
    return perc_days_ice_free


def monthly_mask(ci, time, perc_thresh=30, ci_thresh=0.20):

    # Get month numbers from time array
    months = (time.astype('datetime64[M]').astype(int) % 12) + 1

    # Initialize boolean array for full mask
    full_monthly_mask = np.zeros_like(ci, dtype=bool)

    # Initialize lists to plot boolean mask
    monthly_masks = []

    # Loop through months (all years pooled by month)
    for month in range(1, 13):
        # Get current month's time indices (all years)
        month_indices = months == month

        # Create 2D boolean mask for month where percent ice free days is greater/equal to threshold 
        mask_month = (
            perc_days_ice_free(ci[month_indices], threshold=ci_thresh) >= perc_thresh 
        )

        # Broadcast 2D boolean mask to all time steps for month 
        full_monthly_mask[month_indices, :, :] = mask_month

        # Append 2D boolean mask to list for plotting
        monthly_masks.append(mask_month)

    # Stack list of masks into array
    monthly_masks = np.stack(monthly_masks, axis=0)

    return full_monthly_mask, monthly_masks

def pre_mask_raw_ci(ci_t0):
    """
    Steps taken to mask raw nsidc concentration in mask_normalize step
    """

    # Get NSIDC pre-normalization raw ice conentration
    ci_t0_raw = np.round(ci_t0 * 250)

    # List NSIDC flag values
    nsidc_flags = [
        251, # pole hole
        252, # unused data
        253, # coastline
        254, # land
    ]

    # Mask concentration based on NSIDC flag values
    ci_t0_masked = np.where(
        np.isin(ci_t0_raw, nsidc_flags),
        np.nan,
        ci_t0
    )

    # Get final mask of nans
    ci_nan_mask = np.isnan(ci_t0)

    return ci_t0_masked, ci_nan_mask


def mask_ci(ci_t0, ui_t0, vi_t0, full_monthly_mask, ci_thresh=0.20):
    """
    Steps taken to mask raw nsidc concentration in mask_normalize step
    """

    # Get NSIDC pre-normalization raw ice conentration
    ci_t0_raw = np.round(ci_t0 * 250)

    # List NSIDC flag values
    nsidc_flags = [
        251, # pole hole
        252, # unused data
        253, # coastline
        254, # land
    ]

    # Mask concentration based on NSIDC flag values
    ci_t0_masked = np.where(
        np.isin(ci_t0_raw, nsidc_flags),
        np.nan,
        ci_t0
    )

    # Create mask of bad points
    mask_bad = (
        np.isnan(ci_t0_masked)
        | np.isnan(ui_t0)
        | np.isnan(vi_t0)
        | (ci_t0 <= ci_thresh)
        | full_monthly_mask
    )

    return mask_bad


def plot(monthly_mean, monthly_sem, global_monthly_mean, global_monthly_sem, lat, lon, month_labels, perc_thresh, ci_thresh):

    for ch, ch_name in enumerate(['u', 'v']):

        # -------- Mean plots --------
        plot_cartopy_map(
            data=monthly_mean[:, ch],   # (month, lat, lon)
            lon=lon,
            lat=lat,
            hemisphere=HEMISPHERE,
            titles=month_labels,
            suptitle=f'SKILL MEAN, perc_thresh: {perc_thresh}, ci_thresh: {ci_thresh}({ch_name}): {MODEL_STR}',
            data_channel_axis=0,
            n_cols=4,
            n_rows=3,
            cmap=cmo.cm.balance_r, 
            vmin=-1,
            vmax=1,
            save_path=SAVE_PLOT_PATH / f'mean_{ch_name}_{perc_thresh}_{ci_thresh}.png',
        )

        # -------- SEM plots --------
        plot_cartopy_map(
            data=monthly_sem[:, ch],
            lon=lon,
            lat=lat,
            hemisphere=HEMISPHERE,
            titles=month_labels,
            suptitle=f'SKILL SEM, perc_thresh: {perc_thresh}, ci_thresh: {ci_thresh} ({ch_name}): {MODEL_STR}',
            data_channel_axis=0,
            n_cols=4,
            n_rows=3,
            cmap=cmo.cm.amp,   # better for uncertainty
            vmin=0,
            vmax=np.nanmax(monthly_sem[:, ch]),
            save_path=SAVE_PLOT_PATH / f'sem_{ch_name}_{perc_thresh}_{ci_thresh}.png',
        )

        # -------- Global MEAN/SEM plots --------
        plot_global_monthly_ensemble(
            global_mean=global_monthly_mean,
            global_sem=global_monthly_sem,
            month_labels=month_labels,
            metric_str='SKILL',
            model_str=MODEL_STR,
            save_path=SAVE_PLOT_PATH / f"global_monthly_{perc_thresh}_{ci_thresh}.png",
            title_prefix=f'Global Monthly: perc_thresh: {perc_thresh}, ci_thresh: {ci_thresh}'
            )