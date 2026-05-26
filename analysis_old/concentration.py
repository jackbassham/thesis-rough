import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path


from analysis.plot import plot_cartopy_map, plot_contour_cartopy_map, plot_discrete_cartopy_map
import helpers

ROOT = Path('/data/globus/jbassham/thesis-rough')

HEMISPHERE = 'south'
TIMESTAMP_REGRID = '05062026_1852'
TIMESTAMP_MASK_NORM = TIMESTAMP_REGRID


N_MEMBERS = 10

BASE_PATH = Path(
    ROOT
    / 'regrid'
    / HEMISPHERE
    / TIMESTAMP_REGRID
)


ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')

SAVE_PATH = ANALYSIS_PATH / 'concentration' / HEMISPHERE / 'ci_mean_mask'
SAVE_PATH.mkdir(parents=True, exist_ok=True)

SAVE_MASK_PATH = ANALYSIS_PATH / 'masks' / HEMISPHERE / 'ci_mean_mask'
SAVE_MASK_PATH.mkdir(parents=True, exist_ok=True)




def main():

    # Load and shift ice concentration data
    ci = helpers.load_ice_conc(BASE_PATH, 'ice_conc_regrid_nsidc0051_v2.npz')
    ci_t0 = present_day(ci)

    # Load and shift ice velocity data
    ui, vi, _ = helpers.load_ice_vel(BASE_PATH, 'ice_vel_regrid_nsidc0016_v4.npz')
    ui_t0, vi_t0 = present_day(ui), present_day(vi)

    # Mask using steps taken in mask_normalize
    ci_t0, _ = pre_mask_raw_ci(ci_t0)

    # Load in coordinates
    coord_data = np.load(BASE_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    print('~~~~~~~~~~~Data loaded~~~~~~~~~~~')

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    # # Compute monthly percent days ice free
    # monthly_perc_days_ice_free = monthly_stat(
    #     ci_t0,
    #     time_t0,
    #     perc_days_ice_free,
    #     month_labels = month_labels
    # )

    # # Copute monthly percent days with ice
    # monthly_perc_days_ice = monthly_stat(
    #     ci_t0,
    #     time_t0,
    #     perc_days_ice,
    #     month_labels = month_labels
    # )

    # plot_discrete_cartopy_map(
    #     data=monthly_perc_days_ice,
    #     lon=lon,
    #     lat=lat,
    #     hemisphere=HEMISPHERE,
    #     titles=month_labels,
    #     suptitle='Total Percent Days (100%), (ci<=0.15 & ci>0.15): (1989-2020)',
    #     data_channel_axis=0,
    #     n_cols=4,
    #     n_rows=3,
    #     cmap=cmo.cm.thermal,
    #     cbar_label='%',
    #     boundaries=np.linspace(0,100,num=10),
    #     save_path=Path(SAVE_PATH / "monthly_perc_days_ice.png"),
    # )

    # print(f'ice free shape: {monthly_perc_days_ice_free.shape}')
    # print(f'ice shape: {monthly_perc_days_ice.shape}')

    # # Add monthly percent days with/without ice to verify mask
    # monthly_total_percent = monthly_perc_days_ice_free+monthly_perc_days_ice

    # plot_cartopy_map(
    #     data=monthly_total_percent,
    #     lon=lon,
    #     lat=lat,
    #     hemisphere=HEMISPHERE,
    #     titles=month_labels,
    #     suptitle='Total Percent Days (100%), (ci<=0.15 & ci>0.15): (1989-2020)',
    #     data_channel_axis=0,
    #     n_cols=4,
    #     n_rows=3,
    #     cmap=cmo.cm.thermal,
    #     cbar_label='%',
    #     vmin=0,
    #     vmax=100,
    #     save_path=Path(SAVE_PATH / "monthly_perc_total_days.png"),
    # )

    # Compute monthly ice concntration mean
    monthly_mean = monthly_stat(
        ci_t0,
        time_t0,
        np.nanmean,
        stat_fcn_kwargs={'axis': 0}
    )

    print('~~~~~~~~~~~Stats computed~~~~~~~~~~~')

    # plot_discrete_cartopy_map(
    #     data=monthly_perc_days_ice_free,
    #     lon=lon,
    #     lat=lat,
    #     hemisphere=HEMISPHERE,
    #     titles=month_labels,
    #     suptitle='Percent Days Ice Free, (ci<=0.15): (1989-2020)',
    #     data_channel_axis=0,
    #     n_cols=4,
    #     n_rows=3,
    #     cmap=cmo.cm.thermal,
    #     cbar_label='%',
    #     vmin=0,
    #     vmax=100,
    #     steps=10,
    #     save_path=Path(SAVE_PATH / "monthly_perc_days_ice_free.png"),
    # )

    plot_discrete_cartopy_map(
        data=monthly_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Monthly Mean, ci (Pre-Mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label='frac',
        boundaries=np.arange(0.0, 1.0 + 0.1, 0.1),
        save_path=Path(SAVE_PATH / "ci_monthly_mean_pre_mask_thermal.png"),
    )

    print('~~~~~~~~~~~Pre-mask plots saved~~~~~~~~~~~')

    # Create monthly (pooled accross years) ice concentration mask based on percent days ice free
    full_monthly_mask, monthly_masks = monthly_mask(ci_t0, time_t0)

    # Use monthly mask and additional criteria to mask create total mask of bad points
    mask_bad = mask_ci(ci_t0, ui_t0, vi_t0, full_monthly_mask)

    # Save the mask
    np.savez(
        SAVE_MASK_PATH / 'ci_mean_mask.npz',
        mask_bad = mask_bad
    )

    # Mask bad points to nan
    ci_t0_masked = np.where(mask_bad, np.nan, ci_t0)

    print('~~~~~~~~~~~Masks Created~~~~~~~~~~~')

    # Compute monthly masked ice concentration mean
    monthly_masked_mean = monthly_stat(
        ci_t0_masked,
        time_t0,
        np.nanmean,
        stat_fcn_kwargs={'axis': 0}
    )

    plot_discrete_cartopy_map(
        data=~monthly_masks,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Monthly ci Mask ((monthly_mean_ci </= 0.50)) (inverted)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label='bool ()',
        boundaries=np.arange(0,1+0.5,0.5),
        save_path=Path(SAVE_PATH / "ci_monthly_mask.png"),
    )

    plot_discrete_cartopy_map(
        data=monthly_masked_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Monthly Mean, ci (Post-Mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.ice,
        cbar_label='frac',
        boundaries=np.arange(0.0, 1.0 + 0.1, 0.1),
        save_path=Path(SAVE_PATH / "ci_monthly_mean_post_mask.png"),
    )

    print('~~~~~~~~~~~Post-mask plots saved~~~~~~~~~~~')


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


def perc_days_ice(ci, threshold=0.15):

    # Determine valid, non-nan ice conentration grid points
    valid = ~np.isnan(ci)
    
    # Sum total number of valid days at each gridpoint
    n_total = np.sum(valid, axis=0)

    # Sum number of valid ice free days at each gridpoint
    n_ice_free = np.sum((ci > threshold) & valid, axis=0)

    # Initialize array of nans for percent ice free
    perc_days_ice = np.full_like(n_total, np.nan, dtype=np.float32)

    # Divide where valid points exist, otherwise leave nan
    np.divide(
        n_ice_free * 100,
        n_total,
        out=perc_days_ice,
        where=n_total != 0
    )
            
    return perc_days_ice


def monthly_mask(ci, time, ci_thresh=0.50):

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
            (np.nanmean(ci[month_indices], axis=0)) <= ci_thresh 
        )

        # Broadcast 2D boolean mask to all time steps for month 
        full_monthly_mask[month_indices, :, :] = mask_month

        # Append 2D boolean mask to list for plotting
        monthly_masks.append(mask_month)

    # Stack list of masks into array
    monthly_masks = np.stack(monthly_masks, axis=0)

    return full_monthly_mask, monthly_masks


# def monthly_mask(ci, time, perc_thresh=70, ci_thresh=0.15):

#     # Get month numbers from time array
#     months = (time.astype('datetime64[M]').astype(int) % 12) + 1

#     # Initialize boolean array for full mask
#     full_monthly_mask = np.zeros_like(ci, dtype=bool)

#     # Initialize lists to plot boolean mask
#     monthly_masks = []

#     # Loop through months (all years pooled by month)
#     for month in range(1, 13):
#         # Get current month's time indices (all years)
#         month_indices = months == month

#         # Create 2D boolean mask for month where percent ice free days is greater/equal to threshold 
#         mask_month = (
#             perc_days_ice_free(ci[month_indices], threshold=ci_thresh) >= perc_thresh 
#         )

#         # Broadcast 2D boolean mask to all time steps for month 
#         full_monthly_mask[month_indices, :, :] = mask_month

#         # Append 2D boolean mask to list for plotting
#         monthly_masks.append(mask_month)

#     # Stack list of masks into array
#     monthly_masks = np.stack(monthly_masks, axis=0)

#     return full_monthly_mask, monthly_masks


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


def mask_ci(ci_t0, ui_t0, vi_t0, full_monthly_mask, ci_thresh=0.15):
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


def monthly_stat(data, time, stat_fcn, month_labels=None, stat_fcn_kwargs=None):
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

        if month_labels is not None:
            print(f'~~~~~~~~~~~~~~~{month_labels[i]}~~~~~~~~~~~~~~~')

        # Compute stat for that month, including any keyword arguments
        if stat_fcn_kwargs is not None:
            stat = stat_fcn(data[month_indices], **stat_fcn_kwargs)
        else:
            stat = stat_fcn(data[month_indices])

        # Append to the list of monthly metrics
        monthly_stats.append(stat)

        if month_labels is not None:
            print()

    # Return stacked array of montly metrics along first (month) axis
    return(np.stack(monthly_stats, axis=0)) # (month, height, width)


if __name__ == '__main__':
    main()