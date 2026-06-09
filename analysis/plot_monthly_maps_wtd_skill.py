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

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis3/final_plots/monthly_maps')


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

    # Wtd CNN Zonal Ice
    plot_ensemble_all_models(
        metrics=metrics,
        lon=lon,
        lat=lat,
        channel_idx=0,
        channel_str='u',
        suptitle=r'(a) $u_{i,t}$',
        model_str='cnn_pt_wtd',
        hemisphere=HEMISPHERE,
        save_path=PLOT_PATH,
        figsize=(10,10),
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(-0.8, 0.8+0.1, 0.1)
        )
    
    # Wtd CNN Meridional Ice
    plot_ensemble_all_models(
        metrics=metrics,
        lon=lon,
        lat=lat,
        channel_idx=1,
        channel_str='v',
        suptitle=r'(b) $v_{i,t}$',
        model_str='cnn_pt_wtd',
        hemisphere=HEMISPHERE,
        save_path=PLOT_PATH,
        figsize=(10,10),
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(-0.8, 0.8+0.1, 0.1)
        )


def plot_ensemble_all_models(
        metrics,
        lon,
        lat,
        channel_idx,
        channel_str,
        suptitle,
        model_str,
        hemisphere,
        save_path,
        figsize=(16,3), # (Width, Height)
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(-0.8, 0.8+0.1, 0.1)
    ):

    #~~~~~~~~~~~~~~~~~~~~~
    # Cartopy set up
    #~~~~~~~~~~~~~~~~~~~~~
    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds and projection based on hemisphere
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
    
    # Define data-to-plot's coordinate reference system
    # NOTE, used for 'crs' and 'transform' cartopy parameters
    data_crs = ccrs.PlateCarree()
    data_extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axs = plt.subplots(
        6,
        2,
        figsize=figsize,
        subplot_kw={"projection": projection},
        # constrained_layout=True,
    )

    # Reduce space between map panels
    fig.subplots_adjust(
        left=0.03,
        right=0.88,   # Adjust to leave room for colorbar
        bottom=0.03,
        top=0.90,
        wspace=-0.15,  # Adjust for horizontal space between columns
        hspace=-0.35,  # Adjust for vertical space between rows
    )

    #~~~~~~~~~~~~~~~~~~~~~
    # Panel set up
    #~~~~~~~~~~~~~~~~~~~~~
    panels = [
        ('weighted_skill', 0, axs[0,0], 'Jan'),
        ('weighted_skill', 1, axs[1,0], 'Feb'),
        ('weighted_skill', 2, axs[2,0], 'Mar'),
        ('weighted_skill', 3, axs[3,0], 'Apr'),
        ('weighted_skill', 4, axs[4,0], 'May'),
        ('weighted_skill', 5, axs[5,0], 'Jun'),
        ('weighted_skill', 6, axs[0,1], 'Jul'),
        ('weighted_skill', 7, axs[1,1], 'Aug'),
        ('weighted_skill', 8, axs[2,1], 'Sep'),
        ('weighted_skill', 9, axs[3,1], 'Oct'),
        ('weighted_skill', 10, axs[4,1], 'Nov'),
        ('weighted_skill', 11, axs[5,1], 'Dec'),
    ]

    for metric_str, month_idx, ax, title in panels:

        mean = metrics[model_str][metric_str]['mean'][month_idx][channel_idx]

        ax.set_extent(data_extent, crs=data_crs)
        ax.coastlines()

        gl = ax.gridlines(
            draw_labels=False,
            alpha=0.2,
        )

        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax.set_boundary(circle, transform=ax.transAxes)

        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        pcm = ax.pcolormesh(
            lon,
            lat,
            mean,
            transform=data_crs,
            cmap=cmap,
            norm=norm,
        )

        ax.set_title(
            title, 
            loc='left', 
            fontweight='bold'
            )

    # Set position [left, bottom, width, height] of colorbar in figure coordinates
    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.64])

    cbar = fig.colorbar(
        pcm,
        cax=cbar_ax,
        orientation='vertical',
        label=r'$Skill_w$',
    )

    fig.suptitle(
        suptitle,
        fontweight='bold',
        x=0.05,
        ha='left'
    )

    # vector graphics for publications
    plt.savefig(
        save_path / f'monthly_wtd_skill_{model_str}_{channel_str}.png',
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == '__main__':
    main()

