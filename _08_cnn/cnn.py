import helpers
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
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

    opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4)

    # Define number of epochs
    num_epochs = 50 # Hoffman

    # Initialize losses
    train_losses = []
    val_losses = []

    # Train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_dl, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False):
            # print(torch.isnan(xb).any(), torch.isinf(xb).any())
            opt.zero_grad()
            preds = model(xb)
            loss = NRMSEloss(preds, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        for xb, yb in tqdm(val_dl, desc=f"Val   Epoch {epoch+1}/{num_epochs}", leave=False):
            with torch.no_grad():
                preds = model(xb)
                loss = NRMSEloss(preds, yb)
                val_loss += loss.item()

        avg_train = train_loss / len(train_dl)
        avg_val   = val_loss   / len(val_dl)

        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

    # Load in model outputs destination path
    path_cnn_out = cfg.path_config.model_path('cnn_pt')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_cnn_out)

    # Plot losses
    plot_losses(path_cnn_out, 'cnn_pt_lossses', num_epochs, train_losses, val_losses)

    # Save model weights
    torch.save(
        model.state_dict(), 
        path_cnn_out / 'weights.pt'
        )

    print('Model weights saved')

    # Evaluate trained model
    model.eval()

    # Get predictions on test set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print('y_pred shape:', y_pred.shape)
    print('y_true shape:', y_true.shape)

    # Save predictions and true values
    np.savez(
        path_cnn_out / 'preds.npz', 
        y_pred = y_pred, y_true = y_true)

    print("Predictions saved")


def set_seed(seed=42):
    torch.manual_seed(seed) # PyTorch Reproducibility
    torch.cuda.manual_seed(seed) # Required if using GPU
    torch.backends.cudnn.deterministic = True  # Reproducibility if using GPU
    torch.backends.cudnn.benchmark = False # Paired with above

    return


def plot_losses(path, filename, num_epochs, train_losses, val_losses):
    epochs = np.arange(1, num_epochs + 1)

    
    plt.figure()
    plt.plot(epochs, train_losses, label = 'Train')
    plt.plot(epochs, val_losses, label = 'Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"{MODEL_STR} Loss")

    plt.savefig(path / filename)

    # plt.show()

    return


def NRMSEloss(pred, true, eps=1e-6):
    """
    Norm Root Mean Squared Loss
    """

    mse = torch.mean((pred - true) ** 2)
    std = torch.std(true, unbiased = False) + eps # Unbiased=True To match default pop. std. in tf 

    return torch.sqrt(mse) / std


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)