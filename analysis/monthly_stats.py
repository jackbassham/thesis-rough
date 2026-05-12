import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path


from analysis.plot import plot_cartopy_map, plot_contour_cartopy_map
import helpers

ROOT = Path('/data/globus/jbassham/thesis-rough')

HEMISPHERE = 'south'
TIMESTAMP_REGRID = '05062026_1852'


N_MEMBERS = 10

BASE_PATH = Path(
    ROOT
    / 'regrid'
    / HEMISPHERE
    / TIMESTAMP_REGRID
)

ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')

def main():

    # Load in data variables
    ui, vi, _ = helpers.load_ice_vel(BASE_PATH, 'ice_vel_regrid_nsidc0016_v4.npz')
    ci = helpers.load_ice_conc(BASE_PATH, 'ice_conc_regrid_nsidc0051_v2.npz')

    # Load in coordinates
    coord_data = np.load(BASE_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    print('data loaded')

    # Shift variables to create present day input parameters
    ui_t0, vi_t0 = present_day(ui), present_day(vi)
    ci_t0 = present_day(ci)

    # Compute stats
    monthly_ui_var = monthly_stats(ui_t0, time_t0, np.nanvar, {'axis': 0})
    monthly_vi_var = monthly_stats(ui_t0, time_t0, np.nanvar, {'axis': 0})
    monthly_ci_mean = monthly_stats(ci_t0, time_t0, np.nanmean, {'axis': 0})

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    plot_contour_cartopy_map(
        data=monthly_ui_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ui): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=np.nanmax(monthly_ui_var),
        save_path=Path(ANALYSIS_PATH / "monthly_masked_counts.png"),
    )


def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[:-1,:,:]


def monthly_stats(data, time, stat_fcn, stat_fcn_kwargs):
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

        # Compute stat for that month, including any keyword arguments
        stat = stat_fcn(data[month_indices], **stat_fcn_kwargs)

        # Append to the list of monthly metrics
        monthly_stats.append(stat)

    # Return stacked array of montly metrics along first (month) axis
    return(np.stack(monthly_metrics, axis=0)) # (month, height, width)