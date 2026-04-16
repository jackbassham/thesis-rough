import numpy as np
import torch
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

    x_train, y_train = torch.from_numpy(train['x'][:,:-1,:,:]), torch.from_numpy(train['y'])
    x_val, y_val = torch.from_numpy(val['x'][:,:-1,:,:]), torch.from_numpy(val['y'])
    x_test, y_test = torch.from_numpy(test['x'][:,:-1,:,:]), torch.from_numpy(test['y'])

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
