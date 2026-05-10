import helpers
import numpy as np
import torch
import torch.nn.functional as F
from . import (
    ensemble,
    loss_funcs,
    models,
    utils_cnn
)


MODEL_STR = 'weighted_cnn'

def main(cfg):

    # Set random seed for reproducibility
    helpers.set_seed()

    # Get device
    device = utils_cnn.check_for_accelerator()

    # Load in current member's split targets, features
    splits = ensemble.load_member_splits(cfg)

    # Build tensor datasets from member's splits
    train_ds = utils_cnn.build_dataset(splits['train'], device, include_uncertainty=True, include_mask=True)
    val_ds = utils_cnn.build_dataset(splits['val'], device, include_uncertainty=True, include_mask=True)
    test_ds = utils_cnn.build_dataset(splits['test'], device, include_uncertainty=True, include_mask=True)

    # Get batched data loaders from tensor datasets
    train_dl, val_dl, test_dl = utils_cnn.get_batched_data(train_ds, val_ds, test_ds)

    # Get input dimensions
    in_channels, height, width = utils_cnn.get_input_dimensions(train_dl)

    # Complile model
    # NOTE using PyTorch Default 'Kaiming Uniform' weights/bias initialization
    # Tensorflow Default is Xavier (used by Hoffman)
    # TODO If Kaiming is bad, use function in utils_cnn to apply Xavier initializtion
    model = models.Hoffman_CNN(
        in_channels, height, width).to(device)
    
    # Recursively apply xavier initialization to each layer's weights, set biases to zero
    model.apply(utils_cnn.initialize_weights_biases)

    # Define regularization
    # weight_decay = 1e-4
    # NOTE NOT TRUE: # L2 Norm Regularization, changed from 0.01 in TensorFlow
    # NOTE TensorFLow multiplies Regularization by 0.05 0.01*0.05 -> 5e-4

    # Define Learning Rate
    lr = 1e-4

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
    loss_func = loss_funcs.masked_weighted_nrmse

    # Train the model
    train_losses, val_losses = utils_cnn.fit(epochs, model, loss_func, opt, train_dl, val_dl)

    # Save plot of training losses
    utils_cnn.plot_losses(
        train_losses, val_losses,
        'Weighted CNN',
        timestamp=cfg.version_config.timestamp_model_output,
        member=f'{cfg.runtime.member:02d}',
        path=cfg.path_config.makedir_if_missing(
            cfg.path_config.model_path('cnn_pt_wtd', plot_path=True)
        )
    )

    # Load in model outputs destination path
    # FIXME change naming convention here
    path_cnn_out = cfg.path_config.model_path('cnn_pt_wtd')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_cnn_out)

    # Save weights and biases
    torch.save(
        model.state_dict(), 
        path_cnn_out / 'parameters.pt'
        )

    # Get test predictions as numpy arrays from the trained model
    test_predictions, test_targets = utils_cnn.evaluate(model, test_dl)

    # Save predictions and true values
    np.savez(
        path_cnn_out / 'preds.npz', 
        y_pred = test_predictions, 
        y_true = test_targets
        )

    print("Predictions saved")


if __name__ == "__main__":
    from _00_config.load_config import load_config
    from .ensemble import run_ensemble
    cfg = load_config()
    run_ensemble(cfg, main)