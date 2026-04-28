import cmocean as cmo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from _00_config.parse_args import parse_args

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

    print(f'Shape utrue {np.shape(utrue)}')
    print('')

    # Load path to split indices
    path_inputs = config.path_config.data_stage_path('model_inputs')

    # Get filename for test train split
    fnam = f'split_indices.npz'

    # Load split indices file
    split_indices = np.load(path_inputs / 'split_indices.npz')

    # TODO can get mask for test split from inputs

    # Load path to time variable nan mask (bad points)
    mask_path = config.path_config.data_stage_path('mask_norm')

    # NOTE masking out bad points, not just land/ ocean for plotting

    # Load time variable nan mask file and slice to test split time indices
    mask_bad = np.load(mask_path / 'masks.npz')['mask_bad'][split_indices['test']]

    print(f'Shape mask_bad {np.shape(mask_bad)}')
    print('')

    # # If the model is persistance
    # if model_name == 'ps':
    #     # Shift nan mask one day forward
    #     mask_bad = mask_bad[1:,:,:]

    # Mask invalid points in model output before evaluation
    # NOTE CNN input of 0 at invalid points and LR output of 0 for vi (imag) component

    upred = np.where(mask_bad, np.nan, upred)
    vpred = np.where(mask_bad, np.nan, vpred)

    utrue = np.where(mask_bad, np.nan, utrue)
    vtrue = np.where(mask_bad, np.nan, vtrue)

    # Load in uncertainty
    ri_test = np.load(path_inputs / 'test.npz')['ri_t0']

    # Remove channel dimension from uncertainty
    ri_test = np.squeeze(ri_test, axis = 1)

    # # If the model is persistance
    # # TODO dynamic strings and error conditions
    # if model_name == 'ps':
        
    #     # Shift ri_test array forward one day
    #     ri_test = ri_test[1:,:,:]

    # Load path to coordinates
    path_coordinates = config.path_config.data_stage_path('regrid')

    # Load in lat and lon
    data = np.load(path_coordinates / 'coordinates.npz')
    lon = data['lon']
    lat = data['lat']
    
    # Calculate and plot skill
    plot_metric(
        skill(upred, utrue),
        skill(vpred, vtrue),
        lon,
        lat,
        "Skill",
        path_model, model_name, config,
    )

    print("Skill Plotted")

    # Calculate and plot weighted skill
    plot_metric(
        weighted_skill(upred, utrue, ri_test),
        weighted_skill(vpred, vtrue, ri_test),
        lon,
        lat,
        "Wtd Skill",
        path_model, model_name, config,
    )

    print("Weighted Skill Plotted")
    print("")

    # Calculate and plot correlation
    plot_metric(
        correlation(upred, utrue),
        correlation(vpred, vtrue),
        lon,
        lat,
        "Corr",
        path_model, model_name, config,
    )

    print("Correlation Plotted")
    print("")

    # Calculate and plot correlation
    plot_metric(
        weighted_correlation(upred, utrue, ri_test),
        weighted_correlation(vpred, vtrue, ri_test),
        lon,
        lat,
        "Wtd Corr",
        path_model, model_name, config,
    )

    print("Weighted Correlation Plotted")
    print("")

    print("All Metrics Plotted")


def plot_metric(u_data, v_data, lon, lat, metric, path_model, model_name, config):

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

    # TODO get timestamp version from path
    # NOTE this could be metadata in the model outputs

    # Add title to plot (version specific part of the path)
    fig.suptitle(
        f'{model_name}: {config.version_config.timestamp_model_output}', 
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


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)