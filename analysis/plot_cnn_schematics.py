import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

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
TIMESTAMP_MODEL_INPUTS = TIMESTAMP



PLOT_PATH = Path('/home/jbassham/jack/thesis-rough/analysis/cnn_schematic')


def main():

    path_regrid = Path(
        ROOT
        / 'regrid'
        / HEMISPHERE
        / TIMESTAMP_REGRID
    )

    # Load in coordinate variables
    coordinates = np.load(path_regrid / 'coordinates.npz')
    
    lon = coordinates['lon']
    lat = coordinates['lat']

    # Make base plot directory
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    path_model_inputs = Path(
        ROOT
        / 'model_inputs'
        / HEMISPHERE
        / TIMESTAMP_REGRID
    )

    tf = np.load(path_model_inputs / "targets_features.npz", mmap_mode="r")

    t_idx = 180

    x_day = tf["x"][t_idx]
    y_day = tf["y"][t_idx]

    ui_t0 = y_day[0]
    vi_t0 = y_day[1]

    ua_t0 = x_day[0]
    va_t0 = x_day[1]
    ci_t1 = x_day[2]
    mask_bad = x_day[3]
    

    plot_schematic_feature(
        ui_t0,
        lon,
        lat,
        HEMISPHERE,
        title=None,
        cmap=cmo.cm.curl,
        vmin=-1,
        vmax=1,
        save_path=PLOT_PATH / 'ui_t0.png'
    )

    plot_schematic_feature(
        vi_t0,
        lon,
        lat,
        HEMISPHERE,
        title=None,
        cmap=cmo.cm.curl,
        vmin=-1,
        vmax=1,
        save_path=PLOT_PATH / 'vi_t0.png'
    )

    plot_schematic_feature(
        ci_t1,
        lon,
        lat,
        HEMISPHERE,
        title=None,
        vmin=0,
        vmax=1,
        cmap=cmo.cm.ice,
        save_path=PLOT_PATH / 'ci_t1.png'
    )

    plot_schematic_feature(
        ua_t0,
        lon,
        lat,
        HEMISPHERE,
        title=None,
        cmap=cmo.cm.curl,
        vmin=-1,
        vmax=1,
        save_path=PLOT_PATH / 'ua_t0.png'
    )

    plot_schematic_feature(
        va_t0,
        lon,
        lat,
        HEMISPHERE,
        title=None,
        cmap=cmo.cm.curl,
        vmin=-1,
        vmax=1,
        save_path=PLOT_PATH / 'va_t0.png'
    )

    plot_schematic_feature(
        mask_bad,
        lon,
        lat,
        HEMISPHERE,
        title=None,
        cmap=cmo.cm.gray_r,
        save_path=PLOT_PATH / 'mask.png'
    )


def plot_schematic_feature(
    data,
    lon,
    lat,
    hemisphere,
    title=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    save_path=None,
):
    """
    Plot a feature map for CNN architecture schematics.

    Rectangular map with coastlines only.
    """

    if hemisphere.lower() == "south":
        lat_min, lat_max = -80, -62
    elif hemisphere.lower() == "north":
        lat_min, lat_max = 65, 90
    else:
        raise ValueError('hemisphere must be "north" or "south"')

    fig, ax = plt.subplots(
        figsize=(3, 2),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.pcolormesh(
        lon,
        lat,
        data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )

    ax.set_extent(
        [-180, 180, lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )

    # Fill the axes box instead of preserving geographic aspect ratio
    ax.set_aspect("auto")

    # coastlines 
    ax.coastlines()

    # remove everything else
    ax.set_xticks([])
    ax.set_yticks([])

    # remove border
    ax.spines[:].set_visible(False)

    if title:
        ax.set_title(title)

    plt.tight_layout(pad=0.01)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.01,
        )

    return fig, ax


if __name__ == '__main__':
    main()