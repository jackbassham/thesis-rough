import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np


def plot_cartopy_map(
        data, lon, lat,
        hemisphere: str,
        titles: list[str] | None = None,
        suptitle=None,
        data_channel_axis: int | None = None,
        n_cols=2,
        n_rows=1,
        cmap=cmo.cm.balance_r,
        cbar_label=None,
        vmin=-1,
        vmax=1,
        save_path=None,
):
    
    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds and projection based on hemisphere
    if hemisphere.lower().strip() == 'south':
        lat_min = -90
        lat_max = -65
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

    # If data channel dimension is specified for multiple data in array
    if data_channel_axis is not None:
        # Infer the number of plots from that dimension
        n_plots = data.shape[data_channel_axis]

    # Otherwise number of plots is one
    else:
        n_plots = 1

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    # Flatten axs array for iteration
    axs = np.atleast_1d(axs).flatten()

    for i in range(n_plots):
        ax = axs[i]

        ax.set_extent(data_extent, crs=data_crs)
        ax.coastlines()

        pcm = ax.pcolormesh(
            lon,
            lat,
            data[i],
            transform=data_crs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if titles is not None:
            ax.set_title(titles[i])

    # Turn off unused axes
    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    # Shared colorbar
    fig.colorbar(pcm, ax=axs[:n_plots], orientation="vertical", shrink=0.8, label=cbar_label)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs
    

def plot_contour_cartopy_map(
        data, lon, lat,
        hemisphere: str,
        titles: list[str] | None = None,
        suptitle=None,
        data_channel_axis: int | None = None,
        n_cols=2,
        n_rows=1,
        cmap=cmo.cm.balance_r,
        cbar_label=None,
        vmin=-1,
        vmax=1,
        n_levels=10,
        save_path=None,
):
    
    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds and projection based on hemisphere
    if hemisphere.lower().strip() == 'south':
        lat_min = -90
        lat_max = -65
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

    # If data channel dimension is specified for multiple data in array
    if data_channel_axis is not None:
        # Infer the number of plots from that dimension
        n_plots = data.shape[data_channel_axis]

    # Otherwise number of plots is one
    else:
        n_plots = 1

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    # Flatten axs array for iteration
    axs = np.atleast_1d(axs).flatten()

    levels = np.linspace(vmin, vmax, n_levels)

    for i in range(n_plots):
        ax = axs[i]

        ax.set_extent(data_extent, crs=data_crs)
        ax.coastlines()

        ctf = ax.contourf(
            lon,
            lat,
            data[i],
            levels=levels,
            transform=data_crs,
            cmap=cmap,
        )

        if titles is not None:
            ax.set_title(titles[i])

    # Turn off unused axes
    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    # Shared colorbar
    fig.colorbar(ctf, ax=axs[:n_plots], orientation="vertical", shrink=0.8, label=cbar_label)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs