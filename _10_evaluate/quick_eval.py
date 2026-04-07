import cmocean as cmo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# TODO silence mean of empty slice warning


def main():
    
    ...

    # 
    args = 'insert_arg_parse'

    run_eval(cfg, model_str=args.model)


def run_eval(cfg, model_name: str) -> None:
    """
    
    """

    # Load path to model predictions
    path_model = cfg.path_config.model_path(model_name)

    # Load in model predictions
    data = np.load(path_model / 'preds.npz') 

    upred = data['y_pred'][:,0,:,:]
    vpred = data['y_pred'][:,1,:,:]

    utrue = data['y_true'][:,0,:,:]
    vtrue = data['y_true'][:,1,:,:]

    print(f'Shape utrue {np.shape(utrue)}')
    print('')

    # Load path to split indices
    path_inputs = cfg.path_config.data_stage_path('model_input')

    # Get filename for test train split
    fnam = f'split_indices.npz'

    # Load split indices file
    split_indices = np.load(path_inputs / 'split_indices.npz')

    # TODO can get mask for test split from inputs

    # Load path to time variable nan mask (bad points)
    mask_path = cfg.path_config.data_stage_path('mask_norm')

    # NOTE masking out bad points, not just land/ ocean for plotting

    # Load time variable nan mask file and slice to test split time indices
    mask_bad = np.load(mask_path / 'masks.npz')['mask_bad'][split_indices['test']]

    print(f'Shape mask_bad {np.shape(mask_bad)}')
    print('')

    # If the model is persistance
    if model_name == 'ps':

        # Shift nan mask one day forward
        mask_bad = mask_bad[1:,:,:]

    # Mask invalid points in model output before evaluation
    # NOTE CNN input of 0 at invalid points and LR output of 0 for vi (imag) component

    upred = np.where(mask_bad, np.nan, upred)
    vpred = np.where(mask_bad, np.nan, vpred)

    utrue = np.where(mask_bad, np.nan, utrue)
    vtrue = np.where(mask_bad, np.nan, vtrue)

    # Get filemane for uncertainty test split
    fnam = f"test.npz"

    # Load in uncertainty
    ri_test = np.load(path_inputs / 'test.npz')['ri_t0']

    # If the model is persistance
    # TODO dynamic strings and error conditions
    if model_name == 'ps':
        
        # Shift ri_test array forward one day
        ri_test = ri_test[1:,:,:]

    # Load path to coordinates
    path_coordinates = cfg.path_config.data_stage_path('regrid')

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
        "Skill"
    )

    print("Skill Plotted")

    # Calculate and plot weighted skill
    plot_metric(
        weighted_skill(upred, utrue, ri_test),
        weighted_skill(vpred, vtrue, ri_test),
        lon,
        lat,
        "Wtd Skill"
    )

    print("Weighted Skill Plotted")
    print("")

    # Calculate and plot correlation
    plot_metric(
        correlation(upred, utrue),
        correlation(vpred, vtrue),
        lon,
        lat,
        "Corr"
    )

    print("Correlation Plotted")
    print("")

    # Calculate and plot correlation
    plot_metric(
        weighted_correlation(upred, utrue, ri_test),
        weighted_correlation(vpred, vtrue, ri_test),
        lon,
        lat,
        "Wtd Corr"
    )

    print("Weighted Correlation Plotted")
    print("")

    print("All Metrics Plotted")


def correlation(pred, true):

    """
    Pearson Correlation
    """

    predbar = np.nanmean(pred, axis = 0) # mean predicted
    truebar = np.nanmean(true, axis = 0) # mean true

    covariance = np.nansum((pred - predbar) * (true - truebar), axis = 0) # covariance between predicted and true
    
    stdpred = np.sqrt(np.nansum((pred - predbar)**2, axis = 0)) # standard deviation predited
    stdtrue = np.sqrt(np.nansum((true - truebar)**2, axis = 0)) # standard deviation true

    correlation = covariance / (stdpred * stdtrue)

    return correlation


def weighted_correlation(pred, true, r, epsilon = 1e-4):

    """
    Weighted Pearson Correlation referenced from:
    https://www.air.org/sites/default/files/2021-06/Weighted-and-Unweighted-Correlation-Methods-Large-Scale-Educational-Assessment-April-2018.pdf
    
    """

    w = 1 / (r + epsilon)

    def weighted_mean(x, w):
        return np.nansum(w * x, axis = 0) / np.nansum(w, axis = 0)

    predbar = weighted_mean(pred, w) # weighted mean predicted
    truebar = weighted_mean(true, w) # weighted mean true

    weighted_cov = np.nansum(w * (pred - predbar) * (true - truebar), axis = 0) # weighted covariance between predicted and true
    
    weighted_stdpred = np.sqrt(np.nansum(w * (pred - predbar)**2, axis = 0)) # weighted standard deviation predited
    weighted_stdtrue = np.sqrt(np.nansum(w * (true - truebar)**2, axis = 0)) # weighted standard deviation true

    correlation = weighted_cov / (weighted_stdpred * weighted_stdtrue)

    return correlation


def skill(pred, true, epsilon = 1e-4):
    # NOTE excluding epsilon = 1e-4 from denominator for now

    mse = np.nanmean((true - pred)**2, axis = 0) # mean square error
    # NOTE above is not equivalent to np.nanvar(true-pred), which excludes bias term
    # MSE = E[(y-x)^2]
    # = (E[y-x])^2 + Var(y-x)
    # = bias^2 + Var(y-x)
    # Can prove the above

    truebar = np.nanmean(true, axis = 0) # mean true

    vartrue = np.nanmean((true - truebar)**2, axis = 0) # variance in true
    # NOTE above is equivalent to np.nanvar()

    skill = 1 - mse / (vartrue + epsilon)

    return skill


def weighted_skill(pred, true, r, epsilon = 1e-4):
    # NOTE including epsilon = 1e-4 in the weights in case of uncertainty r ~ 0

    w = 1 / (r + epsilon)

    mse = np.nanmean(( w * (true - pred))**2, axis = 0) # mean square error
    # NOTE above is not equivalent to np.nanvar(true-pred), which excludes bias term

    truebar = np.nanmean(true, axis = 0) # mean true

    vartrue = np.nanmean(( w * (true - truebar))**2, axis = 0) # variance in true
    # NOTE above is equivalent to np.nanvar()

    weighted_skill = 1 - mse / (vartrue + epsilon)

    return weighted_skill


def plot_metric(u_data, v_data, lon, lat, metric, path_model, model_name, cfg):

    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds based on hemisphere
    if cfg.data_config.hemisphere == 'south':
        lat_min = -90
        lat_max = -65
    elif cfg.data_config.hemisphere == 'north':
        lat_min = 65
        lat_max = 90

    # Define plot proection based on hempisphere
    if cfg.data_config.hemisphere == 'south':
        projection = ccrs.SouthPolarStereo()
    elif cfg.data_config.hemisphere == 'north':
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
    fig.suptitle(path_model.split('model_output', maxsplit = 1)[1], fontweight = 'bold')

    # Format with tight layout
    fig.tight_layout

    # Add text with means
    fig.text(0, -0.05, f"mean zonal {metric}: {np.nanmean(u_data):.4f}")
    fig.text(0.5, -0.05, f"mean meridional {metric}: {np.nanmean(v_data):.4f}")

    # Define filemane for figure
    fnam = f"{metric}.png"

    # Load model plot path
    path_plot = cfg.path_config.model_path(model_name, plot_path = True)

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(model_name)

    # Save figure
    plt.savefig(path_plot / fnam, bbox_inches = 'tight')

    return


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)