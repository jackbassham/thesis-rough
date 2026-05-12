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

BASE_NORMALIZED_PATH = Path(
    ROOT
    / 'mask_norm'
    / HEMISPHERE
    / TIMESTAMP_MASK_NORM
)


ANALYSIS_PATH = Path('/home/jbassham/jack/thesis-rough/analysis')

def main():

    # Load in data variables
    ui, vi, ri = helpers.load_ice_vel(BASE_PATH, 'ice_vel_regrid_nsidc0016_v4.npz')
    ci = helpers.load_ice_conc(BASE_PATH, 'ice_conc_regrid_nsidc0051_v2.npz')

    # Load in coordinates
    coord_data = np.load(BASE_PATH / 'coordinates.npz')

    time_t0 = coord_data['time_t0']
    lat = coord_data['lat']
    lon = coord_data['lon']

    print('~~~~~~~~~~~Data loaded~~~~~~~~~~~')

    # Shift variables to create present day input parameters
    ui_t0, vi_t0, ri_t0 = present_day(ui), present_day(vi), present_day(ri)
    ci_t0 = present_day(ci)

    # Compute stats
    monthly_ui_var = monthly_stats(ui_t0, time_t0, var)
    monthly_vi_var = monthly_stats(vi_t0, time_t0, var)
    monthly_ri_mean = monthly_stats(ri_t0, time_t0, np.nanmean, stat_fcn_kwargs={'axis': 0})
    monthly_ci_mean = monthly_stats(ci_t0, time_t0, np.nanmean, stat_fcn_kwargs={'axis': 0})
    monthly_ci_var = monthly_stats(ci_t0, time_t0, var)

    print('~~~~~~~~~~~Stats computed~~~~~~~~~~~')

    # Month labels for titles
    month_labels = [calendar.month_abbr[i+1] for i in range(12)]

    plot_contour_cartopy_map(
        data=monthly_ui_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ui, raw): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=150,
        levels=np.linspace(0,150,num=10),
        save_path=Path(ANALYSIS_PATH / "monthly_raw_var_ui.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_vi_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(vi, raw): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=150,
        levels=np.linspace(0,150,num=10),
        save_path=Path(ANALYSIS_PATH / "monthly_raw_var_vi.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_ri_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Mean(ri, raw): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=100,
        levels=np.linspace(0,100,num=10),
        save_path=Path(ANALYSIS_PATH / "monthly_raw_mean_ri.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_ci_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Mean(ci, raw): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.ice,
        cbar_label='concentration (Frac)',
        vmin=0,
        vmax=1,
        levels=np.arange(0.0, 1.1, 0.1),
        save_path=Path(ANALYSIS_PATH / "monthly_raw_mean_ci.png"),
    )

    plot_cartopy_map(
        data=monthly_ci_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ci, raw): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label='concentration (Frac)',
        vmin=0,
        vmax=np.nanmax(monthly_ci_var),
        save_path=Path(ANALYSIS_PATH / "monthly_raw_var_ci.png"),
    )

    print('~~~~~~~~~~~Plots Saved~~~~~~~~~~~')

    # Load in monthly mask
    mask_bad = np.load(ANALYSIS_PATH / 'monthly_mask.npz')['monthly_mask_bad']
    ui_masked = np.where(mask_bad, np.nan, ui_t0)
    vi_masked = np.where(mask_bad, np.nan, vi_t0)
    ci_masked = np.where(mask_bad, np.nan, ci_t0)
    ri_masked = np.where(mask_bad, np.nan, ri_t0)

    # Compute stats
    monthly_ui_var_masked = monthly_stats(ui_masked, time_t0, var)
    monthly_vi_var_masked = monthly_stats(vi_masked, time_t0, var)
    monthly_ri_mean_masked = monthly_stats(ri_masked, time_t0, np.nanmean, stat_fcn_kwargs={'axis': 0})
    monthly_ci_mean_masked = monthly_stats(ci_masked, time_t0, np.nanmean, stat_fcn_kwargs={'axis': 0})
    monthly_ci_var_masked = monthly_stats(ci_masked, time_t0, var)

    # for m in range(12):
    #     arr = monthly_ci_mean_masked[m]
    #     print(
    #         month_labels[m],
    #         "nan frac:", np.isnan(arr).mean(),
    #         "min:", np.nanmin(arr),
    #         "max:", np.nanmax(arr),
    #         "mean:", np.nanmean(arr),
    #     )

    plot_contour_cartopy_map(
        data=monthly_ui_var_masked,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ui, post mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=150,
        levels=np.linspace(0,150,num=10),
        save_path=Path(ANALYSIS_PATH / "masked_monthly_var_ui.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_vi_var_masked,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(vi, post mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=150,
        levels=np.linspace(0,150,num=10),
        save_path=Path(ANALYSIS_PATH / "masked_monthly_var_vi.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_ri_mean_masked,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Mean(ri, post mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="cm/s",
        vmin=0,
        vmax=100,
        levels=np.linspace(0,100,num=10),
        save_path=Path(ANALYSIS_PATH / "masked_monthly_mean_ri.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_ci_mean_masked,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Mean(ci, post mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.ice,
        cbar_label='concentration (Frac)',
        vmin=0,
        vmax=1,
        levels=np.arange(0.0, 1.1, 0.1),
        save_path=Path(ANALYSIS_PATH / "masked_monthly_mean_ci.png"),
    )

    plot_cartopy_map(
        data=monthly_ci_var_masked,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ci, post mask): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label='concentration (Frac)',
        vmin=0,
        vmax=np.nanmax(monthly_ci_var),
        save_path=Path(ANALYSIS_PATH / "masked_monthly_var_ci.png"),
    )

    data = np.load(BASE_NORMALIZED_PATH / 'masked_normalized.npz')

    ui_t0, vi_t0, ri_t0 = data['ui_t0'], data['vi_t0'], data['ri_t0']
    ci_t1 = data['ci_t1']

    # Compute stats
    monthly_norm_ui_var = monthly_stats(data['ui_t0'], time_t0, var)
    monthly_norm_vi_var= monthly_stats(data['vi_t0'], time_t0, var)
    monthly_norm_ri_mean = monthly_stats(data['ri_t0'], time_t0, np.nanmean, stat_fcn_kwargs={'axis': 0})
    monthly_norm_ci_mean = monthly_stats(data['ci_t1'], time_t0, np.nanmean, stat_fcn_kwargs={'axis': 0})
    monthly_norm_ci_var = monthly_stats(data['ci_t1'], time_t0, var)


    plot_contour_cartopy_map(
        data=monthly_norm_ui_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ui, post mask+norm): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="std (normalized)",
        vmin=0,
        vmax=np.nanmax(monthly_norm_ui_var),
        levels=np.linspace(0,np.nanmax(monthly_norm_ui_var),num=10),
        save_path=Path(ANALYSIS_PATH / "maskednorm_monthly_var_ui.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_norm_vi_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(vi, post mask+norm): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="std (normalized)",
        vmin=0,
        vmax=np.nanmax(monthly_norm_vi_var),
        levels=np.linspace(0,np.nanmax(monthly_norm_vi_var),num=10),
        save_path=Path(ANALYSIS_PATH / "maskednorm_monthly_var_vi.png"),
    )

    plot_contour_cartopy_map(
        data=monthly_norm_ri_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Mean(ri, post mask+norm): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="std (normalized)",
        vmin=np.nanmin(monthly_norm_ri_mean),
        vmax=np.nanmax(monthly_norm_ri_mean),
        levels=np.linspace(np.namin(monthly_norm_ri_mean),np.nanmax(monthly_norm_ri_mean),num=10),
        save_path=Path(ANALYSIS_PATH / "maskednorm_monthly_mean_ri.png"),
    )

    # NOTE contour looks like nan during summer?
    plot_cartopy_map(
        data=monthly_norm_ci_mean,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Mean(ci, post mask+norm): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.ice,
        cbar_label="std (normalized)",
        vmin=np.nanmin(monthly_norm_ci_mean),
        vmax=np.nanmax(monthly_norm_ci_mean),
        levels=np.linspace(np.nanmin(monthly_norm_ci_mean), np.nanmax(monthly_norm_ci_mean), num=10),
        save_path=Path(ANALYSIS_PATH / "maskednorm_monthly_mean_ci.png"),
    )

    plot_cartopy_map(
        data=monthly_norm_ci_var,
        lon=lon,
        lat=lat,
        hemisphere=HEMISPHERE,
        titles=month_labels,
        suptitle='Var(ci, post mask+norm): (1989-2020)',
        data_channel_axis=0,
        n_cols=4,
        n_rows=3,
        cmap=cmo.cm.thermal,
        cbar_label="std (normalized)",
        vmin=0,
        vmax=np.nanmax(monthly_ci_var),
        save_path=Path(ANALYSIS_PATH / "maskednorm_monthly_var_ci.png"),
    )


def var(x):
    xbar = np.nanmean(x, axis = 0) # mean
    return(np.nanmean((x - xbar)**2, axis = 0)) # variance 


def present_day(variable):
    """
    
    """
    return variable[1:,:,:]


def previous_day(variable):
    """
    
    """
    return variable[:-1,:,:]


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


if __name__ == '__main__':
    main()