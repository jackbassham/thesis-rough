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

PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis3/final_plots')


def main():

    # Load in nested dict of metrics 
    metrics = helpers.load_all_metrics(
        ROOT,
        MODEL_STRS,
        METRIC_STRS,
        HEMISPHERE,
        TIMESTAMP,
    )

    u_del_wtd_corr = metrics['cnn_pt_wtd']['weighted_correlation']['mean'][0] - metrics['lr_cf_wtd']['weighted_correlation']['mean'][0]
    v_del_wtd_corr = metrics['cnn_pt_wtd']['weighted_correlation']['mean'][1] - metrics['lr_cf_wtd']['weighted_correlation']['mean'][1]

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
        u_del_wtd_corr,
        v_del_wtd_corr,
        lon,
        lat,
        HEMISPHERE,
        PLOT_PATH,
        figsize=(8,5),
        cmap=cmo.cm.matter,
        boundaries=np.arange(-0.0, 0.35+0.05, 0.05)
    )


def plot_ensemble_all_models(
        u_del_wtd_corr,
        v_del_wtd_corr,
        lon,
        lat,
        hemisphere,
        save_path,
        figsize=(10,8), # (Width, Height)
        cmap=cmo.cm.balance_r,
        boundaries=np.arange(0.0, 1.0+0.1, 0.1)
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
        1,
        2,
        figsize=figsize,
        subplot_kw={"projection": projection},
        # constrained_layout=True,
    )

    # Reduce space between map panels
    fig.subplots_adjust(
        left=0.02,
        right=0.80,   # Adjust to leave room for colorbar
        bottom=0.02,
        top=0.96,
        wspace=0.1,  # Adjust for horizontal space between columns
    )

    #~~~~~~~~~~~~~~~~~~~~~
    # Panel set up
    #~~~~~~~~~~~~~~~~~~~~~
    panels = [
        (u_del_wtd_corr,  axs[0], r'$\Delta \rho_w$ (WCNN, WLR), $u_{i,t}$'),
        (v_del_wtd_corr,  axs[1], r'$\Delta \rho_w$ (WCNN, WLR), $v_{i,t}$'),
    ]

    for delmetric, ax, title in panels:

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
            delmetric,
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
    # cbar_ax = fig.add_axes([0.84, 0.08, 0.025, 0.75])
    cbar_ax = fig.add_axes([0.84, 0.18, 0.025, 0.625])


    cbar = fig.colorbar(
        pcm,
        cax=cbar_ax,
        orientation='vertical',
        label=r'$\Delta \rho_w$',
    )

#     fig.suptitle(
#     r'($\Delta \rho_w$ (WCNN, WLR))',
#     x=0.02,          # left side of figure
#     y=0.99,          # near top
#     ha='left',
#     fontweight='bold',
#     fontsize=14,
# )

    #~~~~~~~~~~~~~~~~
    # NOTE
    # For saving maps: .png, dpi=300
    # BUT try pdf if it's not clear
    # For line plots: .pdf
    #~~~~~~~~~~~~~~~~

    plt.savefig(
        save_path / 'del_wtd_corr.png',
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == '__main__':
    main()

