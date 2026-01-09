import gc
import numpy as np
import os

from .path import (
    PATH_SOURCE,
    PATH_SOURCE_COORD,
    PATH_DEST_LR,
    FSTR_END_IN,
    FSTR_END_OUT,
    FSTR_END_COORD
)

def main():

    # Load in normalized data
    fnam = f'masked_normalized_{FSTR_END_IN}.npz'
    data = np.load(os.path.join(PATH_SOURCE, fnam))

    # Unpack input variables from .npz file
    ui = data['ui']
    vi = data['vi']
    ri = data['ri']
    ua = data['ua']
    va = data['va']
    ci = data['ci']

    print("Input Variables Loaded")

    # Extract time (dates)
    fnam = f"coord_{FSTR_END_COORD}.npz"
    data = np.load(os.path.join(PATH_SOURCE_COORD, fnam), allow_pickle=True)
    time = data['time']

    # Create present day parameters (t0) by shifting forward one day
    ui_t0 = ui[1:,:,:]
    vi_t0 = vi[1:,:,:]
    ua_t0 = ua[1:,:,:]
    va_t0 = va[1:,:,:]
    ri_t0 = ri[1:,:,:]

    # Create present day (t0) time coordinate variable by shifting forward one day
    time_t0 = time[1:]

    # Create previous day parameters (t1) by removing last day
    ci_t1 = ci[:-1,:,:]

    print('Present, Previous day parameters created')

    # Define number of input channels
    n_in = 3

    # Define number of output channels
    n_out = 2

    # Get data dimensions
    nt, nlat, nlon = np.shape(ui_t0) # time, latitude, longitude

    # Initialize feature and target arrays (batch, channels, height, width)
    x = np.zeros((nt, n_in, nlat, nlon)) # Features
    y = np.zeros((nt, n_out, nlat, nlon)) # Targets

    # Fill feature arrays
    x[:, 0, :, :] = ua_t0 # Zonal Wind, present day
    x[:, 1, :, :] = va_t0 # Meridional Wind, present day
    x[:, 2, :, :] = ci_t1 # Ice Concentration, previous day

    # Fill target arrays
    y[:, 0, :, :] = ui_t0 # Zonal Ice Velocity, present day
    y[:, 1, :, :] = vi_t0 # Meridional Ice Velocity, present day

    print("Feature and Target Arrays filled")

    # Reshape uncertainty
    # NOTE DO NOT reshape uncertainty for LR
    # ri_t0 = np.expand_dims(ri_t0, 1)
    # # ri_t0 = ri_t0.unsqueeze(1) # [nt, 1, nlat, nlon]

    years = time_t0.astype('datetime64[Y]').astype(int) + 1970

    # Define split mask based on years
    train_mask = (years >= 1992) & (years <= 2016)
    # val_mask   = (years >= 2017) & (years <= 2018)
    test_mask  = (years >= 2019) & (years <= 2020)

    # Get split indices
    train_idx = np.where(train_mask)[0]
    # val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    # Fill train, validation, and test data arrays
    x_train, y_train, r_train = x[train_idx], y[train_idx], ri_t0[train_idx]
    # x_val, y_val, r_val = x[val_idx], y[val_idx], ri_t0[val_idx]
    x_test, y_test, r_test = x[test_idx], y[test_idx], ri_t0[test_idx]

    # Save splits

    np.savez(
        os.path.join(PATH_DEST_LR, f'train_{FSTR_END_OUT}.npz'),
        x_train = x_train,
        y_train = y_train,
        r_train = r_train
    )

    # np.savez(
    #     os.path.join(PATH_DEST_LR, f'val_{FSTR_END_OUT}.npz'),
    #     x_val = x_val,
    #     y_val = y_val,
    #     r_val = r_val
    # )

    np.savez(
        os.path.join(PATH_DEST_LR, f'test_{FSTR_END_OUT}.npz'),
        x_test = x_test,
        y_test = y_test,
        r_test = r_test
    )

    print(f"Train and Test splits saved at {PATH_DEST_LR}")

    # Save split indices

    np.savez_compressed(
    os.path.join(PATH_DEST_LR, f"split_indices_lr_{FSTR_END_OUT}.npz"),
    train_idx=train_idx,
    # val_idx=val_idx,
    test_idx=test_idx
    )
    
    print(f"Split indices saved at {PATH_DEST_LR}")

    return

if __name__ == "__main__":
    main()