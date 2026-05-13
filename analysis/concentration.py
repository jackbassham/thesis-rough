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
SAVE_PATH.mkdir(patents=True, exis_ok=True)

def main():

    # Load and shift ice concentration data
    ci = helpers.load_ice_conc(BASE_PATH, 'ice_conc_regrid_nsidc0051_v2.npz')
    ci_t0 = present_day(ci)

    # Load in coordinates
    coord_data = np.load(BASE_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    print('~~~~~~~~~~~Data loaded~~~~~~~~~~~')

    # Compute monthly percent days ice free
    monthly_perc_days_ice_free = monthly_stat(
        ci_t0,
        time_t0,
        perc_days_ice_free
    )

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    plot_contour_cartopy_map(
        data=monthly_perc_days_ice_free,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Percent Days Ice Free (ci<=0.15): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label='%',
        vmin=0,
        vmax=100,
        levels=np.arange(0, 110, 10),
        save_path=Path(SAVE_PATH / "monthly_perc_days_ice_free_0.15.png"),
    )


def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[:-1,:,:]    


def perc_days_ice_free(ci, threshold=0.15):

    # Count each gridcell's number of days ice free
    n_ice_free = np.sum(ci <= threshold, axis=0)

    # Return percent days ice free
    return n_ice_free / ci.shape[0] * 100


def monthly_stat(data, time, stat_fcn, stat_fcn_kwargs=None):
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


if __name__ == '__main__':
    main()