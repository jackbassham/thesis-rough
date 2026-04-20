import helpers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 


def check_for_accelerator():
    # Use cuda if available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print('Using CUDA')
    else:
        device = torch.device("cpu")
        print('USING CPU')

    return device


def get_datasets(config, device):
    """
    
    """

    # Load model inputs source path
    path_model_inputs = config.path_config.data_stage_path('model_inputs')

    # Load in input data
    train = np.load(path_model_inputs / 'train.npz')
    val = np.load(path_model_inputs / 'val.npz')
    test = np.load(path_model_inputs / 'test.npz')

    # FIXME debug Double type from model_inputs, (converted to np.float32 in data download)
    x_train, y_train = torch.from_numpy(train['x'][:,:-1,:,:]).float(), torch.from_numpy(train['y']).float()
    x_val, y_val = torch.from_numpy(val['x'][:,:-1,:,:]).float(), torch.from_numpy(val['y']).float()
    x_test, y_test = torch.from_numpy(test['x'][:,:-1,:,:]).float(), torch.from_numpy(test['y']).float()

    # Convert nans to zeros
    x_train, y_train = torch.nan_to_num(x_train), torch.nan_to_num(y_train)
    x_val, y_val = torch.nan_to_num(x_val), torch.nan_to_num(y_val)
    x_test, y_test = torch.nan_to_num(x_test), torch.nan_to_num(y_test)

    # Move tensors to device
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Return tensors wrapped in datasets
    return (
        TensorDataset(x_train, y_train),
        TensorDataset(x_val, y_val),
        TensorDataset(x_test, y_test),
    )


def get_batched_data(
        train_ds, val_ds, test_ds,
        batch_size: int = 365
        ):
    """
    
    """

    # Return batches in DataLoader
    return(
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        # NOTE CHANGED to not shuffling val
        # TODO? Use batch_size*2 for val, like torch nn tutorial?
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,batch_size=batch_size, shuffle=False)
    )


def get_input_dimensions(dataloader):
    """
    
    """
    # Get one batch from the dataloader
    batch = next(iter(dataloader))

    # Get features from the first element of the batch
    xb = batch[0]

    print(f'xb.shape[1:] {xb.shape[1:]}')

    # Get dimensions from batch
    in_channels, height, width = xb.shape[1:]

    return(
        in_channels,
        height,
        width
    )


# Apply Xavier initialization to match TensorFlow default
def initialize_weights_biases(layer):

    # If the layer is convolutional or linear (fully connected)
    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.LazyLinear)):
    # Initilaize that layer's weights to Xavier uniform distribution
        nn.init.xavier_uniform_(layer.weight)

        # Initialize layer has bias
        if layer.bias is not None:
            # Set that layer's bias to zero
            nn.init.zeros_(layer.bias)


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    
    """
    # Compute loss for the batch
    loss = loss_func(model(xb), yb)

    # Perform backprop if an optimizer is passed in (ie: for training set)
    if opt is not None:
        # Reset the gradients from the last iteration
        opt.zero_grad()
        # Compute the gradients for each parameter
        loss.backward()
        # Update the weights, biases using the gradient
        opt.step()
        
    # Return the loss and size of batch
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, val_dl):
    """
    
    """

    # Initialize lists of losses for each epoch
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        # Initialize loss and total number of losses to track training loss
        total_loss = 0
        total_num = 0
        # Iterate through batches and show progress bar
        for xb, yb in tqdm(train_dl, desc=f'Train Epoch {epoch+1}/{epochs}', leave=False):
            # Compute batch's loss and get number samples in batch
            loss, num = loss_batch(model, loss_func, xb, yb, opt=opt)
            # Add batch's loss and number to total
            total_loss += loss * num
            total_num += num
        # Compute average training loss over all batches for epoch
        train_loss = total_loss / total_num

        model.eval()
        # Initialize loss and total number of losses to track validation loss
        total_loss = 0
        total_num = 0
        with torch.no_grad():
            # Iterate through batches and show progress bar
            for xb, yb in tqdm(val_dl, desc=f'Val Epoch {epoch+1}/{epochs}', leave=False):
                # Compute batch's loss and get number of samples in batch
                loss, num = loss_batch(model, loss_func, xb, yb)
                # Add batch's loss and number to total
                total_loss += loss * num
                total_num += num
            # Compute average validation loss over all batches for epoch
            val_loss = total_loss / total_num

        # Append epoch's losses to lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Report epoch's losses
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    print('Training loop complete')

    return train_losses, val_losses
    

def plot_losses(train_losses: list[float], val_losses: list[float], 
                model_name: str, timestamp: str | None = None, path: Path | None = None,) -> None:
    """
    
    """

    # TODO accuracy?

    # Get range of epochs
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label = 'Train')
    plt.plot(epochs, val_losses, label = 'Validation')
    plt.xlabel('Epochs')
    plt.legend()

    title = f'{model_name.title} Loss'

    if timestamp is not None:
        title += timestamp

    plt.title(title)

    if path is not None:
        plt.savefig(path / 'training_losses.png')


def evaluate(model, test_dl):
    """
    
    """

    # TODO 
    predictions = []
    # TODO stop saving targets, use true from 
    # training inputs
    targets = []

    model.eval()

    with torch.no_grad():
        for xb, yb in test_dl:
            # Get batch predictions from forward pass
            pb = model(xb)

            # Appdend batch's predictions to list as numpy array
            predictions.append(pb.cpu().numpy())
            targets.append(pb.cpu().numpy())

        # Return concatentated predictions
        return (
            np.concatenate(predictions, axis=0),
            np.concatenate(targets, axis=0)
        )






