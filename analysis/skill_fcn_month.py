import calendar
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from _00_config.parse_args import parse_args

from _02_regrid.core_regrid import(
    GridSpecs,
    construct_regular_grid,
)

from _05_train_models.ensemble import load_member_splits
import _06_evaluate.metric_fcns as metric_fcns


def main(cfg):

    # Instantiate argument parser
    args = parse_args()

    # Run the evaluation for the particular model
    run_eval(cfg, model_name = args.model_name)


def metric_fcn_month(pred, true, time, metric_fcn, r=None):
    """
    
    """

    # Define number of month bins
    n_months = 12

    # Get month numbers from time array
    months = (time.astype('datetime64[M]').astype(int) % 12) + 1

    # Initialize list for monthly metrics
    monthly_metrics = []

    # Loop through months
    for i in range(12):

        # Get current month's time indices
        month_indices = months == (i + 1)

        # Initialize dict for extra uncertainty keyword argument for weighted metrics
        metric_kwargs = {}

        if r is not None:
            # Add current month's uncertainties array to kword arguments
            metric_kwargs['r'] = r[month_indices]

            # Compute metric for the current month with weighting if 
            month_metric = metric_fcn(
                pred[month_indices],
                true[month_indices],
                **metric_kwargs,
            )

        else:
            # Compute the metric for the current month without weighting
            month_metric = metric_fcn(
                pred[month_indices],
                true[month_indices],
            )

        # Append to the list of monthly metrics
        monthly_metrics.append(month_metric)

    # Return concatenated array of all monthly metrics along 'month' (time) dimension
    return(np.concatenate(monthly_metrics, axis=0))


def plot_cartopy_map(
        data, lon, lat,
        hemisphere: str,
        titles: list[str] | None = None,
        suptitle=None,
        data_channel_axis: int | None = None,
        n_cols=2,
        n_rows=1,
        cmap=cmo.cm.balance_r,
        vmin=-1,
        vmax=1,
        save_path=None,
):
    
    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds and projection based on hemisphere
    if hemisphere.lower.strip() == 'south':
        lat_min = -90
        lat_max = -65
        projection = ccrs.SouthPolarStereo()
    elif hemisphere.lower.strip() == 'north':
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
    fig.colorbar(pcm, ax=axs[:n_plots], orientation="vertical", shrink=0.8)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs


def run_eval(config, model_name: str) -> None:
    """
    
    """

    # Load path to model predictions
    path_model = config.path_config.model_path(model_name)

    # Load in prediction data
    data = np.load(path_model / 'preds.npz')

    # Get predictions from data
    u_pred = data['y_pred'][:,0,:,:]
    v_pred = data['y_pred'][:,1,:,:]

    # Get true values from data
    u_true = data['y_true'][:,0,:,:]
    v_true = data['y_true'][:,1,:,:]

    # Load path to model inputs
    path_inputs = config.path_config.data_stage_path('model_inputs')

    # Load in features from model inputs
    data = np.load(path_inputs / 'test.npz')

    # Get mask for test split from last feature channel
    mask_bad = data['x'][:,-1,:,:]

    # Mask invalid points in model output before evaluation
    upred = np.where(mask_bad, np.nan, upred)
    vpred = np.where(mask_bad, np.nan, vpred)

    utrue = np.where(mask_bad, np.nan, utrue)
    vtrue = np.where(mask_bad, np.nan, vtrue)

    # Get uncertainty from test split and remove channel dimension
    ri_test = np.squeeze(data['ri_t0'], axis=1)

    # Instantiate grid specifications object from config bounds
    grid_specs = GridSpecs(
        lat_bounds = config.data_config.latitude_bounds,
        lon_bounds = config.data_config.longitude_bounds,
        resolution_km = config.data_config.grid_resolution,
    )

    # Construct regular grid to infer data's latitude and longitude for plots
    reg_grid = construct_regular_grid(grid_specs)

    # Get lat/lon variables (FIXME for now)
    lat = reg_grid.lat
    lon = reg_grid.lon

    # Infer time array from config
    time = np.arange(
        np.datetime64(f'{config.data_config.year_range[0]}-01-01'),
        np.datetime64(f'{config.data_config.year_range[1]+1}-01-01'),
        dtype='datetime64[D]'
    )

    hemisphere = config.data_config.hemisphere

    timestamp = config.version_config.timestamp_model_output

    base_suptitle = f'{model_name}: {timestamp}'

    titles = ['Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec']

    # Load model plot path
    path_plot = config.path_config.model_path(model_name, plot_path = True)

    # Make destination directory if missing
    config.path_config.makedir_if_missing(path_plot)

    # Plot zonal correlation
    plot_cartopy_map(
        metric_fcn_month(u_pred, u_true, time, metric_fcns.correlation), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Zonal Correlation ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'zonal_corr.png'
    )

    # Plot meridional correlation
    plot_cartopy_map(
        metric_fcn_month(v_pred, v_true, time, metric_fcns.correlation), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Meridional Correlation ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'meridional_corr.png'
    )

    # Plot zonal skill
    plot_cartopy_map(
        metric_fcn_month(u_pred, u_true, time, metric_fcns.skill), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Zonal Skill ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'zonal_skill.png'
    )

    # Plot meridional skill
    plot_cartopy_map(
        metric_fcn_month(v_pred, v_true, time, metric_fcns.skill), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Meridional Skill ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'meridional_skill.png'
    )

    # Plot zonal weighted correlatoin
    plot_cartopy_map(
        metric_fcn_month(u_pred, u_true, time, metric_fcns.weighted_correlation, r=ri_test), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Zonal Weighted Correlation ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'zonal_wtd_corr.png'
    )

    # Plot meridional weighted correlation
    plot_cartopy_map(
        metric_fcn_month(v_pred, v_true, time, metric_fcns.weighted_correlation, r=ri_test), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Meridional Weighted Correlation ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'meridional_wtd_corr.png'
    )

    # Plot zonal skill
    plot_cartopy_map(
        metric_fcn_month(u_pred, u_true, time, metric_fcns.weighted_skill, r=ri_test), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Zonal Weighted Skill ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'zonal_wtd_skill.png'
    )

    # Plot meridional skill
    plot_cartopy_map(
        metric_fcn_month(v_pred, v_true, time, metric_fcns.weighted_skill, r=ri_test), 
        lon, 
        lat,
        hemisphere,
        titles=titles,
        suptitle='Meridional Weighted Skill ' + base_suptitle,
        n_cols=6,
        n_rows=2,
        save_path=path_plot / 'montly' / 'meridional_wtd_skill.png'
    )