import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path


from analysis.plot import plot_cartopy_map
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

    monthly_mask = mask_monthly(
        ci_t0, ui_t0, vi_t0, time_t0
    )

    np.savez(
        ANALYSIS_PATH / 'monthly_mask',
        monthly_mask = monthly_mask
        )

    print('monthly mask created and saved')

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    plot_cartopy_map(
        data=monthly_mask,   # (month, lat, lon)
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle=f'Monthly Mask',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.balance_r, 
        vmin=-1,
        vmax=1,
        save_path= ANALYSIS_PATH / 'monthly_mask.png',
    )

    print('monthly mask plot saved')





def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[:-1,:,:]


def create_data_masks(
        ci_t0, ui_t0, vi_t0,
        perc_ice_free_threshold=0.70,
        ice_conc_threshold=0.15
) -> tuple:
    """
    NOTE NSIDC considers up to 0.15 ice concentration 'ice free' for ice motion dataset
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
    ci_t0 = np.where(
        np.isin(ci_t0_raw, nsidc_flags),
        np.nan,
        ci_t0
    )

    # Get the number of days from ice concentration
    n_days = ci_t0.shape[0]

    # Count number of days ice free at each gridpoint
    n_ice_free = np.sum(ci_t0 <= ice_conc_threshold, axis = 0)

    # Create mask of nan values at bad data points
    mask_land_ocean = np.all(
        np.isnan(ci_t0)
        | np.isnan(ui_t0)
        | np.isnan(vi_t0)
        | (ci_t0 <= ice_conc_threshold)
        | (n_ice_free > (perc_ice_free_threshold * n_days)),
        axis=0
    )

    return mask_land_ocean


def mask_monthly(
        ci_t0, ui_t0, vi_t0, time
):
    """
    
    """

    # Define number of month bins
    n_months = 12

    # Get month numbers from time array
    months = (time.astype('datetime64[M]').astype(int) % 12) + 1

    # Initialize list for monthly mask
    monthly_mask_land_ocean = []

    # Loop through months
    for i in range(n_months):

        # Get current month's time indices
        month_indices = months == (i + 1)

        mask_land_ocean = create_data_masks(
            ci_t0=ci_t0[month_indices],
            ui_t0=ui_t0[month_indices],
            vi_t0=vi_t0[month_indices],
        )

        # Append to the list of monthly metrics
        monthly_mask_land_ocean.append(mask_land_ocean)

    # Return stacked array of montly metrics along first (month) axis
    return(np.stack(monthly_mask_land_ocean, axis=0)) # (month, height, width)


if __name__ == '__main__':
    main()