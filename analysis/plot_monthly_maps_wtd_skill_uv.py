import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

from analysis import plot_fcns, helpers


MODEL_STRS = ['ps', 'lr_cf', 'lr_cf_wtd', 'cnn_pt', 'cnn_pt_wtd']

METRIC_STRS = [
    'skill',
    'weighted_skill',
    'correlation',
    'weighted_correlation',
    ]

ROOT = Path('/data/globus/jbassham/thesis-rough')

HEMISPHERE = 'south'
TIMESTAMP = '06082026_1154'
TIMESTAMP_REGRID = TIMESTAMP
N_MEMBERS = 10

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis3/final_plots/monthly_maps2')


def main():

    # Load in nested dict of metrics 
    metrics = helpers.load_all_metrics(
        ROOT,
        MODEL_STRS,
        METRIC_STRS,
        HEMISPHERE,
        TIMESTAMP,
        monthly=True,
    )

    # Load in coordinate variables
    coordinates = np.load(
        ROOT
        / 'regrid'
        / HEMISPHERE
        / TIMESTAMP_REGRID
        / 'coordinates.npz'
    )
    
    lon = coordinates['lon']
    lat = coordinates['lat']

    # Make base plot directory
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    plot_ensemble_all_models(
        metrics=metrics,
        metric_str='weighted_correlation',
        model_str='cnn_pt_wtd',
        lon=lon,
        lat=lat,
        suptitle=r'$\rho_w$ (WCNN)',
        hemisphere=HEMISPHERE,
        save_path=PLOT_PATH,
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(-0.8, 0.8+0.1, 0.1)
    )

def plot_ensemble_all_models(
        metrics,
        metric_str,
        model_str,
        lon,
        lat,
        suptitle,
        hemisphere,
        save_path,
        figsize=(12,14),
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(-0.8, 0.8+0.1, 0.1)
    ):

    lon_min = -180
    lon_max = 180

    if hemisphere.lower().strip() == 'south':
        lat_min = -90
        lat_max = -62
        projection = ccrs.SouthPolarStereo()
    elif hemisphere.lower().strip() == 'north':
        lat_min = 65
        lat_max = 90
        projection = ccrs.NorthPolarStereo()
    else:
        raise ValueError('Enter hemisphere argument as string "south" or "north"')

    data_crs = ccrs.PlateCarree()
    data_extent = [lon_min, lon_max, lat_min, lat_max]

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        6, 5,
        width_ratios=[1, 1, 0.08, 1, 1],
        left=0.01,
        right=0.88,
        bottom=0.03,
        top=0.95,
        wspace=0.05,
        hspace=0.12
    )

    axs = np.empty((6, 4), dtype=object)

    for row in range(6):
        for col in range(4):
            gs_col = col if col < 2 else col + 1
            axs[row, col] = fig.add_subplot(
                gs[row, gs_col],
                projection=projection
            )

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    pcm = None

    for month_idx, month in enumerate(months):

        if month_idx < 6:
            row = month_idx
            base_col = 0
        else:
            row = month_idx - 6
            base_col = 2

        for channel_idx, channel_label in zip([0, 1], [r'$u_{i,t}$', r'$v_{i,t}$']):

            ax = axs[row, base_col + channel_idx]

            mean = metrics[model_str][metric_str]['mean'][month_idx][channel_idx]

            ax.set_extent(data_extent, crs=data_crs)
            ax.coastlines()

            ax.gridlines(
                draw_labels=False,
                alpha=0.2,
            )

            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

            pcm = ax.pcolormesh(
                lon,
                lat,
                mean,
                transform=data_crs,
                cmap=cmap,
                norm=norm,
            )

            ax.set_title(
                channel_label,
                loc='center',
                fontweight='bold',
                fontsize=12,
                pad=2
            )

        # Shared bold month label for the u/v pair
        left_ax = axs[row, base_col]
        right_ax = axs[row, base_col + 1]

        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()

        x_mid = 0.5 * (left_pos.x0 + right_pos.x1)
        y_top = max(left_pos.y1, right_pos.y1)

        fig.text(
            x_mid,
            y_top + 0.004,
            month,
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )

    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.64])

    cbar = fig.colorbar(
        pcm,
        cax=cbar_ax,
        orientation='vertical',
        label=r'$\rho_w$',
    )

    fig.text(
        0.01,
        0.985,
        suptitle,
        ha='left',
        va='top',
        fontweight='bold',
        fontsize=16
    )

    plt.savefig(
        save_path / f'monthly_{metric_str}_{model_str}_uv.png',
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == '__main__':
    main()

