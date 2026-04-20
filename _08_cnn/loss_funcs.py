import torch

def nrmse(input, target, eps=1e-4):

    # NOTE # Unbiased=True To match default population std. in tf 
    return torch.sqrt(torch.mean((input - target) ** 2)) / (torch.std(input, unibased = False) + eps)


def weighted_mse(input, target, uncertainty, eps = 1e-6):

    # Compute weights
    w = 1 / (uncertainty + eps)
    
    # Compute weighted square error
    wse = w * (input - target)**2

    # Return weighted mean square error
    return torch.sum(wse) / (torch.sum(w) + eps)


# TODO 
# def weighted_nrmse():
    