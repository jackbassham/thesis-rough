import numpy as np

def weighted_mean(x, w):
    return np.nansum(w * x, axis = 0) / np.nansum(w, axis = 0)

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


def weighted_correlation(pred, true, r, epsilon_wts=1e-4):

    """
    Weighted Pearson Correlation referenced from:
    https://www.air.org/sites/default/files/2021-06/Weighted-and-Unweighted-Correlation-Methods-Large-Scale-Educational-Assessment-April-2018.pdf
    
    """

    w = 1 / (r**2 + epsilon_wts)

    predbar = weighted_mean(pred, w) # weighted mean predicted
    truebar = weighted_mean(true, w) # weighted mean true

    weighted_cov = np.nansum(w * (pred - predbar) * (true - truebar), axis = 0) # weighted covariance between predicted and true
    
    weighted_stdpred = np.sqrt(np.nansum(w * (pred - predbar)**2, axis = 0)) # weighted standard deviation predited
    weighted_stdtrue = np.sqrt(np.nansum(w * (true - truebar)**2, axis = 0)) # weighted standard deviation true

    correlation = weighted_cov / (weighted_stdpred * weighted_stdtrue)

    return correlation


def skill(pred, true, epsilon_var=1):
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

    # print(f'Using VarTrue (NOTE print is mean over grid) {np.nanmean(vartrue)}')

    skill = 1 - mse / (vartrue + (epsilon_var)**2)

    return skill


def weighted_skill(pred, true, r, epsilon_wts=1e-4, epsilon_var=1):
    # NOTE including epsilon = 1e-4 in the weights in case of uncertainty r ~ 0

    w = 1 / (r**2 + epsilon_wts)

    wsum = np.nansum(w, axis=0)

    wmse = np.nansum( w * (true - pred) ** 2, axis = 0) / wsum # weighted mean square error

    wtruebar = np.nansum(w * true, axis = 0) / wsum # weighted mean true

    wvartrue = np.nansum( w * (true - wtruebar) ** 2, axis = 0) / wsum # weighted variance in true
    # NOTE above is equivalent to np.nanvar()

    weighted_skill = 1 - wmse / (wvartrue + epsilon_var)

    return weighted_skill


# def weighted_skill(pred, true, r, epsilon = 1):
#     # NOTE including epsilon = 1e-4 in the weights in case of uncertainty r ~ 0

#     w = 1 / (r**2 + epsilon)

#     mse = np.nanmean( w * (true - pred) ** 2, axis = 0) # mean square error
#     # NOTE above is not equivalent to np.nanvar(true-pred), which excludes bias term

#     truebar = np.nanmean(true, axis = 0) # mean true

#     vartrue = np.nanmean( w * (true - truebar) ** 2, axis = 0) # variance in true
#     # NOTE above is equivalent to np.nanvar()

#     weighted_skill = 1 - mse / (vartrue + epsilon)

#     return weighted_skill


def rmse(pred, true):

    """
    Root Mean Square Error
    """

    mse = np.nanmean((true - pred)**2, axis = 0) # mean square error

    rmse = np.sqrt(mse)

    return rmse


def weighted_mse(pred, true, r, epsilon_wts=1e-4):
    """

    """
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Compute weights
    w = 1 / (r**2 + epsilon_wts)
    
    # Compute weighted square error
    wse = w * (pred - true)**2

    # Return weighted mean square error
    return np.nansum(wse) / (np.nansum(w) + epsilon_wts)


def weighted_rmse(pred, true, r, epsilon_wts=1e-4):
    """
    # NOTE rmse puts units back 
    """
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Compute weights
    w = 1 / (r**2 + epsilon_wts)
    
    # Compute weighted square error
    wse = w * (pred - true)**2

    wmse = np.nansum(wse, axis = 0) / (np.nansum(w, axis = 0) + epsilon_wts)

    return np.sqrt(wmse)


def nrmse(pred, true, epsilon = 1e-4):

    """
    Normalized Root Mean Square Error
    """

    mse = np.nanmean((true - pred)**2, axis = 0) # mean square error

    truebar = np.nanmean(true, axis = 0) # mean true

    vartrue = np.nanmean((true - truebar)**2, axis = 0) # variance in true

    nrmse = np.sqrt(mse) / (np.sqrt(vartrue) + epsilon)

    return nrmse


def mae(pred, true):

    """
    Mean Absolute Error
    """

    mae = np.nanmean(np.abs(true - pred), axis = 0)

    return mae


def mean_misfit(pred, true):

    """
    Mean Misfit
    """

    misfit = np.nanmean(true - pred, axis = 0)

    return misfit