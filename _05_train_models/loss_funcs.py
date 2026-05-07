import torch

def nanstd(x, eps=1e-8):
    """
    Equivalent to torch.std(x,unbiased=False) using torch.nanmean to avoid nans
    """
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2) + eps)


def nrmse(input, target, eps=1e-4):

    # NOTE # Unbiased=True To match default population std. in tf 
    return torch.sqrt(torch.mean((input - target) ** 2)) / (torch.std(target, unbiased = False) + eps)


def masked_nrmse(input, target, mask, eps=1e-4):

    # Compute square error 
    se = (input - target) ** 2

    # Set invalid points in square error to nan so they don't contribute to loss
    se = torch.where(mask, torch.nan, se)

    # Set invalid points in target to nan so they don't contribute to loss
    target_masked = torch.where(mask, torch.nan, target)

    rmse = torch.sqrt(torch.nanmean(se))
    norm = nanstd(target_masked)

    return rmse / (norm + eps)


def weighted_mse(input, target, uncertainty, eps = 1e-6):
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Compute weights
    w = 1 / (uncertainty**2 + eps)
    
    # Compute weighted square error
    wse = w * (input - target)**2

    # Return weighted mean square error
    return torch.sum(wse) / (torch.sum(w) + eps)

def weighted_nrmse(input, target, uncertainty, eps = 1e-6):
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Compute weights
    w = 1 / (uncertainty**2 + eps)
    
    # Compute weighted square error
    wse = w * (input - target)**2

    # Compute weighted mean square error
    mse = torch.sum(wse) / (torch.sum(w) + eps)

    # Return the normalized root mean square error
    return torch.sqrt(mse) / nanstd(target) + eps


def masked_weighted_nrmse(input, target, uncertainty, mask, eps = 1e-6):
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Compute weights
    w = 1 / (uncertainty**2 + eps)
    
    # Compute weighted square error
    wse = w * (input - target)**2

    # Set invalid points to nan
    wse = torch.where(mask, torch.nan, wse)
    w = torch.where(mask, torch.nan, wse)

    # Compute weighted mean square error
    mse = torch.nansum(wse) / (torch.nansum(w) + eps)

    # Set invalid points in target to nan
    target_masked = torch.where(mask, torch.nan, target)

    # Return the normalized root mean square error
    return torch.sqrt(mse) / (torch.nanstd(target_masked, unbiased = False) + eps)

# TODO 
# def weighted_nrmse():
    