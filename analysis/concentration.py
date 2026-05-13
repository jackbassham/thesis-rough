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

SAVE_PATH = ANALYSIS_PATH / 'monthly_stats'


def main():

    ...


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