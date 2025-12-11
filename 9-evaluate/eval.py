import cmocean as cmo
import gc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

TIMESTAMP_IN = os.getenv("TIMESTAMP_IN") # timestamp version of input data

TIMESTAMP_MODEL = os.getenv("TIMESTAMP_MODEL") # timestamp version of model run

MODEL_STR = os.getenv("MODEL_STR") # string indicator for model type

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define path to model outputs
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir,
        '..',
        'data',
        HEM,
        TIMESTAMP_MODEL
    )
)



def main():

    # Get filename, based on the model string given

    # Load in true and predicted model outputs
    data = np.load(os.path.join(PATH_SOURCE, fnam)) 

    upred = data['y_pred'][:,0,:,:]
    vpred = data['y_pred'][:,1,:,:]

    utrue = data['y_true'][:,0,:,:]
    vtrue = data['y_true'][:,1,:,:]

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

def plot_metric(u_data, v_data, model_str):

    # Set longitude bounds for plot (full zonal coverage)
    lon_min = -180
    lat_min = 180

    # Set latitude bounds based on hemisphere
    if HEM == 'sh':
        lat_min = -90
        lat_max = -65
    elif HEM =='nh':
        lat_min = 65
        lat_max = 90
    else:
        



