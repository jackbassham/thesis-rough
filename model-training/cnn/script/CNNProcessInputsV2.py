import torch
import numpy as np
import os

####################################################
# Same as original, just changes source to inputs_v2 
# (for uncertainty without mean removed)
####################################################

START_YEAR = 1992
END_YEAR = 2020
HEM = 'sh'

PATH_SOURCE = "/home/jbassham/jack/data/sh/inputs_v2"
PATH_DEST = PATH_SOURCE

# Set random seed for reproducibility

def set_seed(seed=42):
    torch.manual_seed(seed) # PyTorch Reproducibility
    torch.cuda.manual_seed(seed) # Required if using GPU
    torch.backends.cudnn.deterministic = True  # Reproducibility if using GPU
    torch.backends.cudnn.benchmark = False # Paired with above

    return

def main():

    # Set random seed for reproducibility
    set_seed(42)

    # Load input variable file
    fnam = f'inputs_normalized_{HEM}_{START_YEAR}_{END_YEAR}.npz'
    data = np.load(os.path.join(PATH_SOURCE, fnam))

    # Unpack input variables from .npz file
    uit = data['uitn']
    vit = data['vitn']
    rt = data['rtn']
    uwt = data['uwtn']
    vwt = data['vwtn']
    icy = data['icyn']

    # Pack input variables into list
    invars = [uit, vit, rt, uwt, vwt, icy]

    print("Input Variables Loaded")

    # Get landmask (where always nan) (USE LATER FOR PLOTTING!)
    land_mask = np.all(np.isnan(uit), axis = 0)

    # Convert NaN values in inputs to zero
    invars = [np.nan_to_num(var, 0) for var in invars]

    # Unpack variables
    uit, vit, rt, uwt, vwt, icy = invars

    print("Input NaNs Converted to 0")

    # Extract time (dates)
    fnam = f"time_today_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    tt = data['time_today']

    print("Time Loaded")

    # Get dimensions
    nt, ny, nx = np.shape(uit) # time, latitude, longitude

    # Define number of input channels
    n_in = 3

    # Define number of output channels
    n_out = 2

    # Initialize PyTorch Tensors (batch, channels, height, width)
    x = torch.zeros((nt, n_in, ny, nx), dtype = torch.float32) # Features
    y = torch.zeros((nt, n_out, ny, nx), dtype = torch.float32) # Targets

    # Fill feature arrays
    x[:, 0, :, :] = torch.from_numpy(uwt) # Zonal Wind, Today
    x[:, 1, :, :] = torch.from_numpy(vwt) # Meridional Wind, Today
    x[:, 2, :, :] = torch.from_numpy(icy) # Ice Concentration, Yesterday

    # Fill target arrays
    y[:, 0, :, :] = torch.from_numpy(uit) # Zonal Ice Velocity, Today
    y[:, 1, :, :] = torch.from_numpy(vit) # Meridional Ice Velocity, Today

    print("Feature and Target Arrays filled")

    # Reshape uncertainty
    r = torch.from_numpy(rt)
    r = r.unsqueeze(1) # [nt, 1, ny, nx]

    years = tt.astype('datetime64[Y]').astype(int) + 1970

    # Define ranges
    train_mask = (years >= 1992) & (years <= 2016)
    val_mask   = (years >= 2017) & (years <= 2018)
    test_mask  = (years >= 2019) & (years <= 2020)

    # Get indices
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    # Fill train, validation, and test data arrays
    x_train, y_train, r_train = x[train_idx], y[train_idx], r[train_idx]
    x_val, y_val, r_val = x[val_idx], y[val_idx], r[val_idx]
    x_test, y_test, r_test = x[test_idx], y[test_idx], r[test_idx]

    # Save data splits
    pnam = f"cnn_inputs"
    save_path = os.path.join(PATH_DEST, pnam)

    # Make directory if it dosn't exist
    os.makedirs(save_path, exist_ok = True)

    fstr = f"{HEM}_{START_YEAR}_{END_YEAR}"

    torch.save((x_train, y_train, r_train), os.path.join(save_path, f'train_{fstr}.pt'))
    torch.save((x_val, y_val, r_val), os.path.join(save_path, f'val_{fstr}.pt'))
    torch.save((x_test, y_test, r_test), os.path.join(save_path, f'test_{fstr}.pt'))

    print(f"Train, Validation, and Test splits saved at {save_path}")

    np.savez_compressed(
    os.path.join(save_path, f"indices_land_{fstr}.npz"),
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    land_mask = land_mask
)
    
    print(f"Train, Validation, and Test indices saved at {save_path}")

    return

if __name__ == "__main__":
    main()




