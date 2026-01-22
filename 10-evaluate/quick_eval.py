import cmocean as cmo
import gc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO silence mean of empty slice warning

from .param import(
    MODEL_STR,
    MODEL_DIR,
    TIMESTAMP_MODEL,
    HEM,
)

from .path import(
    PATH_SOURCE,
    PATH_DEST,
    PATH_COORD,
    PATH_R,
    FSTR_END_MODEL,
    FSTR_END_COORD,
    FSTR_END_R,
)

def main():

    # Get filename for model predictions, based on the model string given
    fnam = f"preds_{MODEL_STR}_{FSTR_END_MODEL}.npz"

    # Load in model predictions
    data = np.load(os.path.join(PATH_SOURCE, fnam)) 

    upred = data['y_pred'][:,0,:,:]
    vpred = data['y_pred'][:,1,:,:]

    utrue = data['y_true'][:,0,:,:]
    vtrue = data['y_true'][:,1,:,:]

    # Get filemane for uncertainty test split
    fnam = f"test_{FSTR_END_R}.npz"

    # Load in uncertainty
    data = np.load(os.path.join(PATH_R, fnam))

    # Ice Velocity Uncertainty
    # Includes flag values (r_raw + 1000, cm/s)
    # Normalized by std ice speed
    r_test = data['r_test']

    # If the model is persistance
    # TODO dynamic strings and error conditions
    if MODEL_STR == 'ps':
        # Shift r_test array forward one day
        r_test = r_test[1:,:,:]

    # Get filename for lat lon coordinate variables
    fnam = f"coord_{FSTR_END_COORD}.npz"

    # Load in lat and lon
    data = np.load(os.path.join(PATH_COORD, fnam))
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
        weighted_skill(upred, utrue, r_test),
        weighted_skill(vpred, vtrue, r_test),
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
        weighted_correlation(upred, utrue, r_test),
        weighted_correlation(vpred, vtrue, r_test),
        lon,
        lat,
        "Wtd Corr"
    )

    print("Weighted Correlation Plotted")
    print("")

    print("All Metrics Plotted")
    
    return


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

    skill = 1 - mse / vartrue

    return skill

def weighted_skill(pred, true, r, epsilon = 1e-4):
    # NOTE excluding epsilon = 1e-4 from denominator for now
    # NOTE including epsilon = 1e-4 in the weights in case of uncertainty r ~ 0

    w = 1 / (r + epsilon)

    mse = np.nanmean(( w * (true - pred))**2, axis = 0) # mean square error
    # NOTE above is not equivalent to np.nanvar(true-pred), which excludes bias term

    truebar = np.nanmean(true, axis = 0) # mean true

    vartrue = np.nanmean(( w * (true - truebar))**2, axis = 0) # variance in true
    # NOTE above is equivalent to np.nanvar()

    weighted_skill = 1 - mse / vartrue

    return weighted_skill

def plot_metric(u_data, v_data, lon, lat, metric):

    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lon_max = 180

    # Set latitude bounds based on hemisphere
    if HEM == 'sh':
        lat_min = -90
        lat_max = -65
    elif HEM =='nh':
        lat_min = 65
        lat_max = 90

    # Define plot proection based on hempisphere
    if HEM == 'sh':
        projection = ccrs.SouthPolarStereo()
    elif HEM == 'nh':
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
    axs[0].coastlines
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
    axs[1].coastlines
    # Plot pcolormesh plot
    pcm_1 = axs[1].pcolormesh(
        lon, lat, v_data,
        transform = crs,
        cmap = cmap, vmin = vmin, vmax = vmax
    )
    axs[1].set_title("meridional")
    # Add colorbar
    plt.colorbar(pcm_1, ax = axs[1], orientation = 'vertical')

    # Add title to plot
    fig.suptitle(f"{metric}; {MODEL_STR.upper()} v{TIMESTAMP_MODEL}", fontweight = 'bold')

    # Format with tight layout
    fig.tight_layout

    # Add text with means
    fig.text(0, -0.05, f"mean zonal {metric}: {np.nanmean(u_data):.4f}")
    fig.text(0.5, -0.05, f"mean meridional {metric}: {np.nanmean(v_data):.4f}")

    # Define filemane for figure
    fnam = f"{metric}_{MODEL_STR}_{TIMESTAMP_MODEL}.png"

    # Save figure
    plt.savefig(os.path.join(PATH_DEST, fnam), bbox_inches = 'tight')

    return

if __name__ == "__main__":
    main()