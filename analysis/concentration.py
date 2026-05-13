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

SAVE_PATH = ANALYSIS_PATH / 'concentration' / HEMISPHERE
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def main():

    # Load and shift ice concentration data
    ci = helpers.load_ice_conc(BASE_PATH, 'ice_conc_regrid_nsidc0051_v2.npz')
    ci_t0 = present_day(ci)

    # Mask using steps taken in mask_normalize
    ci_t0, ci_nan_mask = mask_ci(ci_t0)

    # Load in coordinates
    coord_data = np.load(BASE_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    print('~~~~~~~~~~~Data loaded~~~~~~~~~~~')

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    # Compute monthly percent days ice free
    monthly_perc_days_ice_free = monthly_stat(
        ci_t0,
        time_t0,
        perc_days_ice_free,
        month_labels = month_labels
    )

    # Compute monthly percent days ice free
    monthly_mean = monthly_stat(
        ci_t0,
        time_t0,
        np.nanmean,
        stat_fcn_kwargs={'axis': 0}
    )

    print('~~~~~~~~~~~Stats computed~~~~~~~~~~~')

    plot_discrete_cartopy_map(
        data=monthly_perc_days_ice_free,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Percent Days Ice Free, (ci<=0.15): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label='%',
        vmin=0,
        vmax=100,
        steps=10,
        save_path=Path(SAVE_PATH / "monthly_perc_days_ice_free_0.15_2.png"),
    )

    plot_discrete_cartopy_map(
        data=monthly_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Monthly Mean, ci: (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.ice,
        cbar_label='frac',
        vmin=0.0,
        vmax=1.0,
        steps=0.1,
        save_path=Path(SAVE_PATH / "ci_monthly_mean.png"),
    )

    print('~~~~~~~~~~~Plots saved~~~~~~~~~~~')



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


# def perc_days_ice_free(ci, threshold=0.15):

#     print(f'ci_shape: {ci.shape}')

#     # Determine valid, non-nan ice conentration grid points
#     valid = ~np.isnan(ci)

#     print(f'valid_shape: {valid.shape}')
    
#     # Sum total number of valid days at each gridpoint
#     n_total = np.sum(valid, axis=0)

#     # Sum number of valid ice free days at each gridpoint
#     n_ice_free = np.sum((ci <= threshold) & valid, axis=0)

#     ny = ci.shape[1]
#     nx = ci.shape[2]

#     # Initialize array for percent ice free
#     perc_days_ice_free = np.full((ny, nx), np.nan)

#     for iy in range(ny):
#         for ix in range(nx):

#             if n_total[iy,ix] != 0:
#                 perc_days_ice_free[iy,ix] = n_ice_free[iy,ix] / n_total[iy,ix] * 100

#             else:
#                 pass
            
#     return perc_days_ice_free


# def perc_days_ice_free(ci, threshold=0.15):

#     # Count each gridcell's number of days ice free
#     n_ice_free = np.sum(ci <= threshold, axis=0)


#     return n_ice_free / ci.shape[0] * 100


def mask_ci(ci_t0):
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