import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path 

MODEL_STRS = ['cnn_pt', 'cnn_pt_wtd', 'lr_cf', 'lr_cf_wtd', 'ps']

ROOT = Path('/data/globus/jbassham/thesis-rough')
MODEL_STR = MODEL_STRS[0]
HEMISPHERE = 'north'
TIMESTAMP = '05012026_1003'

TIMESTAMP_REGRID = TIMESTAMP

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
    plot_path = plot_path_base / MODEL_STR / HEMISPHERE / TIMESTAMP / 'ensemble_mean'
    
    # Make plot path if it doesn't yet exist
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load lat lon coordinates
    data = np.load(ROOT / 'regrid' / HEMISPHERE / TIMESTAMP_REGRID / 'coordinates.npz')

    lat = data['lat']
    lon = data['lon']

    for metric_str in metric_strs:

        u, v = load_member_metrics(metric_str)

        u_mean = np.nanmean(u, axis=0)
        v_mean = np.nanmean(v, axis=0)

        u_sem = np.nanstd(u, axis=0) / np.sqrt(N_MEMBERS)
        v_sem = np.nanstd(v, axis=0) / np.sqrt(N_MEMBERS)

        save_path = BASE_PATH

        np.savez(
            save_path / f'ensemble_mean_{metric_str}.npz',
            u=u_mean,
            v=v_mean,
            u_sem=u_sem,
            v_sem=v_sem,
        )

        plot_metric(
            u_mean, v_mean, lon, lat, HEMISPHERE, 
            ['zonal', 'merdional'], 
            f'Ensemble Mean {metric_str}: {MODEL_STR} {TIMESTAMP}',
            plot_path / f'{metric_str}_mean.png',
        )

        plot_metric(
            u_sem, v_sem, lon, lat, HEMISPHERE, 
            ['zonal', 'merdional'], 
            f'Ensemble SEM {metric_str}: {MODEL_STR} {TIMESTAMP}',
            plot_path / f'{metric_str}_sem.png',
            vmin = -0.05, vmax = 0.05
        )

        print(f'{MODEL_STR}: {metric_str} mean and mse plotted')
        print('')

    print(f'All metrics plotted')


def load_member_metrics(metric_str):
    

    # Initialize lists for metrics
    u_metrics = []
    v_metrics = []

    for m in range(N_MEMBERS):

        path_member = BASE_PATH / f'member_{m:02d}'

        metric_data = np.load(path_member / f'metric_{metric_str}.npz')

        u_metrics.append(metric_data['u'])
        v_metrics.append(metric_data['v'])

    # Stack members in list into array along member axis
    u_metrics_arr = np.stack(u_metrics, axis = 0)
    v_metrics_arr = np.stack(v_metrics, axis = 0)

    return u_metrics_arr, v_metrics_arr


def plot_metric(u_data, v_data, lon, lat, hemisphere, titles, suptitle, plot_path, vmin=-1, vmax=1):

    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds based on hemisphere
    if hemisphere == 'south':
        lat_min = -90
        lat_max = -65
    elif hemisphere == 'north':
        lat_min = 65
        lat_max = 90

    # Define plot proection based on hempisphere
    if hemisphere == 'south':
        projection = ccrs.SouthPolarStereo()
    elif hemisphere == 'north':
        projection = ccrs.NorthPolarStereo()

    # Define data-to-plot's coordinate reference system
    # NOTE, used for 'crs' and 'transform' cartopy parameters
    crs = ccrs.PlateCarree()

    # Set color map
    cmap = cmo.cm.balance_r  # red blue colormap from cmocean
    
    # Initialize subplots
    fig, axs = plt.subplots(
        nrows = 1,
        ncols = 2,
        figsize = (6,3),
        subplot_kw = {'projection': projection},
        constrained_layout = True
    )

    # Plot left plot; zonal evaluation
    axs[0].set_extent([lon_min, lon_max, lat_min, lat_max], crs = crs)
    axs[0].coastlines()
    # Plot pcolormesh plot
    pcm_0 = axs[0].pcolormesh(
        lon, lat, u_data,
        transform = crs,
        cmap = cmap, vmin = vmin, vmax = vmax
    )
    axs[0].set_title(titles[0])
    # Add colorbar
    plt.colorbar(pcm_0, ax = axs[0], orientation = 'vertical')

    # Plot right plot; meridional evaluation
    axs[1].set_extent([lon_min, lon_max, lat_min, lat_max], crs = crs)
    axs[1].coastlines()
    # Plot pcolormesh plot
    pcm_1 = axs[1].pcolormesh(
        lon, lat, v_data,
        transform = crs,
        cmap = cmap, vmin = vmin, vmax = vmax
    )
    axs[1].set_title(titles[1])
    # Add colorbar
    plt.colorbar(pcm_1, ax = axs[1], orientation = 'vertical')

    # Add title to plot (version specific part of the path)
    fig.suptitle(suptitle, fontweight = 'bold')

    # Format with tight layout
    fig.tight_layout

    # Add text with means
    fig.text(0, -0.05, f'{np.nanmean(u_data):.4f}')
    fig.text(0.5, -0.05, f'{np.nanmean(v_data):.4f}')

    # Save figure
    plt.savefig(plot_path, bbox_inches = 'tight')

    return


if __name__ == '__main__':
    main()