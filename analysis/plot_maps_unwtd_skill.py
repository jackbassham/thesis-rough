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
TIMESTAMP = '05222026_1652'
TIMESTAMP_REGRID = TIMESTAMP
N_MEMBERS = 10

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/final_plots/maps_unwtd_metrics_unwtd')


def main():

    # Load in nested dict of metrics 
    metrics = helpers.load_all_metrics(
        ROOT,
        MODEL_STRS,
        METRIC_STRS,
        HEMISPHERE,
        TIMESTAMP,
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
        metrics,
        lon,
        lat,
        HEMISPHERE,
        PLOT_PATH,
        figsize=(10,10),
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(-0.8, 0.8+0.1, 0.1)
        )


def plot_ensemble_all_models(
        metrics,
        lon,
        lat,
        hemisphere,
        save_path,
        figsize=(10,10), # (Width, Height)
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
        3,
        4,
        figsize=figsize,
        subplot_kw={"projection": projection},
        # constrained_layout=True,
    )

    # Reduce space between map panels
    fig.subplots_adjust(
        left=0.05,
        right=1.0,   # Adjust to leave room for colorbar
        bottom=0.04,
        top=0.96,
        wspace=0.1,  # Adjust for horizontal space between columns
        hspace=0.1,  # Adjust for vertical space between rows
    )

    #~~~~~~~~~~~~~~~~~~~~~
    # Panel set up
    #~~~~~~~~~~~~~~~~~~~~~
    panels = [
        ('cnn_pt', 'skill', 0, axs[0,0], r'(a) $u_{i,t}$'),
        ('cnn_pt', 'skill', 1, axs[0,1], r'(a) $v_{i,t}$'),
        ('cnn_pt', 'weighted_skill', 0, axs[0,2], r'(d) $u_{i,t}$'),
        ('cnn_pt', 'weighted_skill', 1, axs[0,3], r'(d) $v_{i,t}$'),
        ('lr_cf', 'skill', 0, axs[1,0], r'(b) $u_{i,t}$'),
        ('lr_cf', 'skill', 1, axs[1,1], r'(b) $v_{i,t}$'),
        ('lr_cf', 'weighted_skill', 0, axs[1,2], r'(e) $u_{i,t}$'),
        ('lr_cf', 'weighted_skill', 1, axs[1,3], r'(e) $v_{i,t}$'),
        ('ps', 'skill', 0, axs[2,0], r'(c) $u_{i,t}$'),
        ('ps', 'skill', 1, axs[2,1], r'(c) $v_{i,t}$'),
        ('ps', 'weighted_skill', 0, axs[2,2], r'(f) $u_{i,t}$'),
        ('ps', 'weighted_skill', 1, axs[2,3], r'(f) $v_{i,t}$'),
    ]

    for model_str, metric_str, channel, ax, title in panels:

        mean = metrics[model_str][metric_str]['mean'][channel]

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
    cbar_ax = fig.add_axes([1.05, 0.18, 0.025, 0.64])

    cbar = fig.colorbar(
        pcm,
        cax=cbar_ax,
        orientation='vertical',
        label=r'$Skill$',
    )

    # vector graphics for publications
    plt.savefig(
        save_path / 'unwtd_skill.png',
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == '__main__':
    main()

