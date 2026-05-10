import torch

def nanstd(x, eps=1e-8):
    """
    Equivalent to torch.std(x,unbiased=False) using torch.nanmean to avoid nans
    """
    return torch.sqrt(torch.nanmean((x - torch.nanmean(x))**2) + eps)


def nanstd(x):
    """
    Equivalent to torch.std(x, unbiased=False), ignoring NaNs.
    """
    mean = torch.nanmean(x)
    var = torch.nanmean((x - mean) ** 2)
    return torch.sqrt(var)


def masked_nrmse(input, target, mask, eps=1e-6):

    # Make sure mask broadcasts over target channels
    mask = mask.bool()

    # Use tensor-valued NaNs
    nan_input = torch.full_like(input, torch.nan)
    nan_target = torch.full_like(target, torch.nan)

    # Compute squared error
    se = (input - target) ** 2

    # Mask invalid locations
    se_masked = torch.where(mask, nan_input, se)
    target_masked = torch.where(mask, nan_target, target)

    rmse = torch.sqrt(torch.nanmean(se_masked))
    norm = nanstd(target_masked)


    return rmse / (norm + eps)



# def nrmse(input, target, eps=1e-4):

#     # NOTE # Unbiased=True To match default population std. in tf 
#     return torch.sqrt(torch.mean((input - target) ** 2)) / (torch.std(target, unbiased = False) + eps)


# def masked_nrmse(input, target, mask, eps=1e-6):

    # # Check mask
    # print("mask shape:", mask.shape)
    # print("target shape:", target.shape)

    # print("masked fraction:", mask.float().mean())

    # print(
    #     "valid count:",
    #     (~mask.expand_as(target)).sum()
    # )

#     # Compute square error 
#     se = (input - target) ** 2

#     # Set invalid points in square error to nan so they don't contribute to loss
#     se = torch.where(mask, torch.nan, se)

#     # Set invalid points in target to nan so they don't contribute to loss
#     target_masked = torch.where(mask, torch.nan, target)

    # # Check cound finite targets
    # print(
    #     "number finite targets_masked:",
    #     torch.isfinite(target_masked).sum()
    # )

#     rmse = torch.sqrt(torch.nanmean(se))
#     norm = nanstd(target_masked)

    # print("rmse:", rmse)
    # print("norm:", norm)

#     return rmse / (norm + eps)


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
    return torch.sqrt(mse) / (nanstd(target) + eps)


def masked_weighted_nrmse(input, target, uncertainty, mask, eps = 1e-6):
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Make sure mask broadcasts over target channels
    mask = mask.bool()

    # print("mask shape:", mask.shape)
    # print("mask true fraction:", mask.float().mean())
    # print("valid count:", (~mask).sum())


    if mask.ndim == input.ndim - 1:
        mask = mask[:, None, :, :]

    if uncertainty.ndim == input.ndim - 1:
        uncertainty = uncertainty[:, None, :, :]

    # Compute weights
    w = 1 / (uncertainty**2 + eps)

    # Match input/target shape
    mask = mask.expand_as(input)
    w = w.expand_as(input)

    # Use tensor-valued NaNs
    nan_input = torch.full_like(input, torch.nan)
    nan_target = torch.full_like(target, torch.nan)
    nan_w = torch.full_like(w, torch.nan)
    
    # Compute weighted square error
    se = (input - target)**2
    wse = w * se
    wse = w * (input - target)**2

    # Mask invalid locations
    wse_masked = torch.where(mask, nan_input, wse)
    target_masked = torch.where(mask, nan_target, target)
    w_masked = torch.where(mask, nan_w, w)

    # Compute weighted mean square error
    wmse = torch.nansum(wse_masked) / (torch.nansum(w_masked) + eps)
    norm = nanstd(target_masked)
    wrmse = torch.sqrt(wmse) / (norm + eps)

    # print("wmse:", wmse)
    # print("norm:", norm)
    # print("rmse:", wrmse)

    # Return the normalized, weighted root mean square error
    return wrmse


# TODO 
# def weighted_nrmse():
    