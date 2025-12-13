import gc
import torch
import numpy as np
import os

from .path import (
    PATH_SOURCE_MASKNORM,
    PATH_SOURCE_COORD,
    PATH_DEST_CNN,
    FSTR_END_IN_MASKNORM,
    FSTR_END_IN_COORD,
    FSTR_END_OUT,
)

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

    # Load in normalized data
    fnam = f'masked_normalized_{FSTR_END_IN_MASKNORM}.npz'
    data = np.load(os.path.join(PATH_SOURCE_MASKNORM, fnam))

    # Unpack input variables from .npz file
    ui = data['ui']
    vi = data['vi']
    ri = data['ri']
    ua = data['ua']
    va = data['va']
    ci = data['ci']

    print("Input Variables Loaded")

    # Convert NaN values in inputs to zero
    ui_filt = np.nan_to_num(ui, 0)
    vi_filt = np.nan_to_num(vi, 0)
    ua_filt = np.nan_to_num(ua, 0)
    va_filt = np.nan_to_num(va, 0)
    ci_filt = np.nan_to_num(ci, 0)

    print("Input NaNs Converted to 0")

    # Convert NaN values in uncertainty to 1000 (flag)
    ri_filt = np.where(np.isnan(ri), 1e3, ri)

    print("Uncertainty NaNs Converted to 1000")

    # Delete arrays to free memory
    del ui, vi, ua, va, ci, ri
    gc.collect() 

    # Extract time (dates)
    fnam = f"coordinates_{FSTR_END_IN_COORD}.npz"
    data = np.load(os.path.join(PATH_SOURCE_COORD, fnam), allow_pickle=True)
    time = data['time']

    # Create present day parameters (t0) by shifting forward one day
    ui_t0 = ui_filt[1:,:,:]
    vi_t0 = vi_filt[1:,:,:]
    ua_t0 = ua_filt[1:,:,:]
    va_t0 = va_filt[1:,:,:]
    ri_t0 = ri_filt[1:,:,:]

    # Create present day (t0) time coordinate variable by shifting forward one day
    time_t0 = time[1:]

    # Create previous day parameters (t1) by removing last day
    ci_t1 = ci_filt[:-1,:,:]

    print('Present, Previous day parameters created')

    # Define number of input channels
    n_in = 3

    # Define number of output channels
    n_out = 2

    # Get data dimensions
    nt, nlat, nlon = np.shape(ui_t0) # time, latitude, longitude

    # Initialize PyTorch Tensors (batch, channels, height, width)
    x = torch.zeros((nt, n_in, nlat, nlon), dtype = torch.float32) # Features
    y = torch.zeros((nt, n_out, nlat, nlon), dtype = torch.float32) # Targets

    # Fill feature arrays
    x[:, 0, :, :] = torch.from_numpy(ua_t0) # Zonal Wind, present day
    x[:, 1, :, :] = torch.from_numpy(va_t0) # Meridional Wind, present day
    x[:, 2, :, :] = torch.from_numpy(ci_t1) # Ice Concentration, previous day

    # Fill target arrays
    y[:, 0, :, :] = torch.from_numpy(ui_t0) # Zonal Ice Velocity, present day
    y[:, 1, :, :] = torch.from_numpy(vi_t0) # Meridional Ice Velocity, present day

    print("Feature and Target Arrays filled")

    # Reshape uncertainty
    ri_t0 = torch.from_numpy(ri_t0)
    ri_t0 = ri_t0.unsqueeze(1) # [nt, 1, ny, nx]

    years = time_t0.astype('datetime64[Y]').astype(int) + 1970

    # Define split mask based on years
    train_mask = (years >= 1992) & (years <= 2016)
    val_mask   = (years >= 2017) & (years <= 2018)
    test_mask  = (years >= 2019) & (years <= 2020)

    # Get split indices
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    # Fill train, validation, and test data arrays
    x_train, y_train, r_train = x[train_idx], y[train_idx], ri_t0[train_idx]
    x_val, y_val, r_val = x[val_idx], y[val_idx], ri_t0[val_idx]
    x_test, y_test, r_test = x[test_idx], y[test_idx], ri_t0[test_idx]

    # Save splits

    torch.save(
        (x_train, y_train, r_train), 
        os.path.join(PATH_DEST_CNN, f'train_{FSTR_END_OUT}.pt')
        )
    
    torch.save(
        (x_val, y_val, r_val), 
        os.path.join(PATH_DEST_CNN, f'val_{FSTR_END_OUT}.pt')
        )

    torch.save(
        (x_test, y_test, r_test), 
        os.path.join(PATH_DEST_CNN, f'test_{FSTR_END_OUT}.pt')
        )

    print(f"Train, Validation, and Test splits saved at {PATH_DEST_CNN}")

    # Save split indices

    np.savez_compressed(
    os.path.join(PATH_DEST_CNN, f"split_indices_cnn_{FSTR_END_OUT}.npz"),
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
    )
    
    print(f"Split indices saved at {PATH_DEST_CNN}")

    return

if __name__ == "__main__":
    main()




