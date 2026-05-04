import cmocean as cmo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from _00_config.parse_args import parse_args
from _02_regrid.core_regrid import(
    GridSpecs,
    construct_regular_grid,
)
from _05_train_models.ensemble import load_member_splits
from . import metric_fcns

# TODO silence mean of empty slice warning


def main(cfg):
    
    # Instantiate argumnet parser
    args = parse_args()

    # Run model 
    run_eval(cfg, model_name = args.model_name)


def run_eval(config, model_name: str) -> None:
    """
    
    """

    # Load path to model predictions
    path_model = config.path_config.model_path(model_name)

    # Load in model predictions
    data = np.load(path_model / 'preds.npz') 

    upred = data['y_pred'][:,0,:,:]
    vpred = data['y_pred'][:,1,:,:]

    utrue = data['y_true'][:,0,:,:]
    vtrue = data['y_true'][:,1,:,:]

    # Load current ensemble splits
    splits = load_member_splits(config)

    # Get mask for test split from last feature channel
    mask_bad = splits['test']['x'][:,-1,:,:]

    # Mask invalid points in model output before evaluation
    upred = np.where(mask_bad, np.nan, upred)
    vpred = np.where(mask_bad, np.nan, vpred)

    utrue = np.where(mask_bad, np.nan, utrue)
    vtrue = np.where(mask_bad, np.nan, vtrue)

    # Get uncertainty from test split and remove channel dimension
    ri_test = np.squeeze(splits['test']['ri_t0'], axis=1)

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

    compute_and_plot_metric(
        upred, utrue, 
        vpred, vtrue,
        metric_fcns.skill,
        'skill',
        model_name,
        lon, lat,
        path_model,
        config,
    )

    print('Skill Plotted')
    print('')
    
    compute_and_plot_metric(
        upred, utrue, 
        vpred, vtrue,
        metric_fcns.weighted_skill,
        'wtd_skill',
        model_name,
        lon, lat,
        path_model,
        config,
        r = ri_test
    )

    print('Weighted Skill Plotted')
    print('')

    compute_and_plot_metric(
        upred, utrue, 
        vpred, vtrue,
        metric_fcns.correlation,
        'corr',
        model_name,
        lon, lat,
        path_model,
        config,
    )

    print('Correlation Plotted')
    print('')

    compute_and_plot_metric(
        upred, utrue, 
        vpred, vtrue,
        metric_fcns.weighted_correlation,
        'wtd_corr',
        model_name,
        lon, lat,
        path_model,
        config,
        r = ri_test
    )

    print('Weighted Correlation Plotted')
    print('')


def plot_metric(u_data, v_data, lon, lat, metric, model_name, config):

    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds based on hemisphere
    if config.data_config.hemisphere == 'south':
        lat_min = -90
        lat_max = -65
    elif config.data_config.hemisphere == 'north':
        lat_min = 65
        lat_max = 90

    # Define plot proection based on hempisphere
    if config.data_config.hemisphere == 'south':
        projection = ccrs.SouthPolarStereo()
    elif config.data_config.hemisphere == 'north':
        projection = ccrs.NorthPolarStereo()

    # Define data-to-plot's coordinate reference system
    # NOTE, used for 'crs' and 'transform' cartopy parameters
    crs = ccrs.PlateCarree()

    # Set color map
    cmap = cmo.cm.balance_r  # red blue colormap from cmocean

    # Saturate colormap to -1 and 1 limits
    vmin = -1
    vmax = 1
    
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
    axs[0].set_title("zonal")
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
    axs[1].set_title("meridional")
    # Add colorbar
    plt.colorbar(pcm_1, ax = axs[1], orientation = 'vertical')

    # Get ensemble member number
    m = config.runtime.member

    # Add title to plot (version specific part of the path)
    fig.suptitle(
        f'{metric} k={m:02d} {model_name}: {config.version_config.timestamp_model_output}', 
        fontweight = 'bold')

    # Format with tight layout
    fig.tight_layout

    # Add text with means
    fig.text(0, -0.05, f"mean zonal {metric}: {np.nanmean(u_data):.4f}")
    fig.text(0.5, -0.05, f"mean meridional {metric}: {np.nanmean(v_data):.4f}")

    # Define filemane for figure
    fnam = f"{metric}.png"

    # Load model plot path
    path_plot = config.path_config.model_path(model_name, plot_path = True)

    # Make destination directory if missing
    config.path_config.makedir_if_missing(path_plot)

    # Save figure
    plt.savefig(path_plot / fnam, bbox_inches = 'tight')

    return


def compute_and_plot_metric(
        u_pred, u_true, 
        v_pred, v_true,
        metric_fcn,
        metric_str,
        model_name,
        lon, lat,
        path_model,
        config,
        r = None,
):
    """

    """

    # Initialize dict for extra uncertainty keyword argument for weighted metrics
    metric_kwargs = {}


    if r is not None:
        # Add current month's uncertainties array to kword arguments
        metric_kwargs['r'] = r
        
        # Compute metric for the current month with weighting 
        u_metric = metric_fcn(u_pred, u_true, **metric_kwargs)
        v_metric = metric_fcn(v_pred, v_true, **metric_kwargs)

    else:
        # Compute the metric for the current month without weighting
        u_metric = metric_fcn(u_pred, u_true)
        v_metric = metric_fcn(v_pred, v_true)

    # Save the zonal and meridional metric
    np.savez(
        path_model / f'metric_{metric_str}.npz',
        u = u_metric,
        v = v_metric
    )

    # Plot the zonal and meridional metrics and save
    plot_metric(
        u_metric,
        v_metric,
        lon,
        lat,
        metric_str,
        model_name, 
        config
    )




    



if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)