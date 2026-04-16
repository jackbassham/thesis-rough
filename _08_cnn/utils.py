import helpers
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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