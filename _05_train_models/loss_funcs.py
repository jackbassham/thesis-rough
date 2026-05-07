import torch

def nrmse(input, target, eps=1e-4):

    # NOTE # Unbiased=True To match default population std. in tf 
    return torch.sqrt(torch.mean((input - target) ** 2)) / (torch.std(target, unbiased = False) + eps)


def weighted_mse(input, target, uncertainty, eps = 1e-6):
    # NOTE must think about w = 1 / (uncertainty**2 + eps) to match weighted linear regression 
    # Weighted mse is used for the closed form solution!

    # Compute weights
    w = 1 / (uncertainty**2 + eps)
    
    # Compute weighted square error
    wse = w * (input - target)**2

    # Return weighted mean square error
    return torch.sum(wse) / (torch.sum(w) + eps)


# TODO 
# def weighted_nrmse():
    