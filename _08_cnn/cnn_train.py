import helpers
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from . import loss_funcs
from . import models
from . import utils

"""
NOTE forward shapes from print statements
layer 1: torch.Size([365, 7, 40, 261])
layer 2: torch.Size([365, 14, 20, 130])
layer 3: torch.Size([365, 28, 10, 65])
layer 4: torch.Size([365, 56, 5, 32])
layer 5: torch.Size([365, 112, 2, 16])
dropout layer: torch.Size([365, 112, 2, 16])
flatten: torch.Size([365, 3584])
fully connected: torch.Size([365, 84564])
outputs: torch.Size([365, 2, 81, 522])
"""

MODEL_STR = 'cnn'

def main(cfg):

    # Set random seed for reproducibility
    helpers.set_seed()

    # Get device
    device = utils.check_for_accelerator()

    # Load tensor datasets from numpy arrays
    train_ds, val_ds, test_ds = utils.get_datasets(cfg, device)

    # Get batched data loaders from tensor datasets
    train_dl, val_dl, test_dl = utils.get_batched_data(train_ds, val_ds, test_ds)

    # Get input dimensions
    in_channels, height, width = utils.get_input_dimensions(train_dl)

    # Complile model
    # NOTE using PyTorch Default 'Kaiming Uniform' weights/bias initialization
    # Tensorflow Default is Xavier (used by Hoffman)
    # TODO If Kaiming is bad, use function in utils to apply Xavier initializtion
    model = models.Hoffman_CNN(
        in_channels, height, width).to(device)
    
    # Recursively apply xavier initialization to each layer's weights, set biases to zero
    model.apply(utils.initialize_weights_biases)

    # Define regularization
    weight_decay = 1e-2 
    # weight_decay = 1e-4
    # NOTE NOT TRUE: # L2 Norm Regularization, changed from 0.01 in TensorFlow
    # NOTE TensorFLow multiplies Regularization by 0.05 0.01*0.05 -> 5e-4

    # Define Learning Rate
    lr = 1e-4

    for name, param in model.named_parameters():
        print(f'name: {name}')
        print(f'param: {param}')

    # Initialize optimizer with weight decay (l2 regularization)
    # NOTE try AdamW for weight decay similar to tf kernel weight regularization
    #https://discuss.pytorch.org/t/how-to-implement-pytorch-equivalent-of-keras-kernel-weight-regulariser/99773
    # https://arxiv.org/abs/1711.05101
    # 

    # Define the optimizer
    opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4)

    # Define the number of training epochs
    epochs = 50 # Hoffman

    # Define the loss function
    loss_func = loss_funcs.nrmse

    # Train the model
    train_losses, val_losses = utils.fit(epochs, model, loss_func, opt, train_dl, val_dl)

    # Save plot of training losses
    utils.plot_losses(
        train_losses, val_losses,
        'CNN',
        timestamp=cfg.version_congig.timestamp_model_output,
        path=cfg.path_config.model_path('cnn_pt', plot_path=True)
    )

    # Load in model outputs destination path
    path_cnn_out = cfg.path_config.model_path('cnn_pt')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_cnn_out)

    # Save weights and biases
    torch.save(
        model.state_dict(), 
        path_cnn_out / 'parameters.pt'
        )

    # Get test predictions as numpy arrays from the trained model
    test_predictions, test_targets = utils.evaluate(model, test_dl)

    # Save predictions and true values
    np.savez(
        path_cnn_out / 'preds.npz', 
        y_pred = test_predictions, 
        y_true = test_targets
        )

    print("Predictions saved")


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)