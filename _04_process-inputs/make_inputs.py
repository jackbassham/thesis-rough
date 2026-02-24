import numpy as np
import os

from _00_config.path import(
    PATH_MASK_NORM,
    PATH_COORDINATES,
    PATH_MODEL_INPUTS,
)

def main():

    # Load in normalized data
    fnam = 'masked_normalized.npz'
    data = np.load(os.path.join(PATH_MASK_NORM, fnam))

    # Unpack input variables from .npz file
    ui_t0 = data['ui_t0']
    vi_t0 = data['vi_t0']
    ri_t0 = data['ri_t0']
    ua_t0 = data['ua_t0']
    va_t0 = data['va_t0']
    ci_t1 = data['ci_t1']

    print("Input Variables Loaded")

    # Load in nan mask
    fnam = 'nan_mask.npz'
    data = np.load(os.path.join(PATH_MASK_NORM, fnam))

    nan_mask = data['nan_mask']

    # Initialize nan mask input, where mask = 1 where valid and 0 where data is nan
    nan_mask_input = np.ones_like(nan_mask)

    # Convert indices where nans exist to 0
    nan_mask_input = np.where(np.isnan(nan_mask), 0, nan_mask_input)

    # Define number of input channels
    n_in = 4

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
    x[:, 3, :, :] = nan_mask_input

    # Fill target arrays
    y[:, 0, :, :] = ui_t0 # Zonal Ice Velocity, present day
    y[:, 1, :, :] = vi_t0 # Meridional Ice Velocity, present day

    print("Feature and Target Arrays filled")

    # Reshape uncertainty
    ri_t0 = np.expand_dims(ri_t0, axis = 1) # [nt, 1, nlat, nlon]
    
    # Extract time (dates)
    fnam = f"coordinates.npz"
    data = np.load(os.path.join(PATH_COORDINATES, fnam), allow_pickle=True)
    # time_t0 = data['time_t0']

    # TODO remove the below, new coordinates file does this
    time = data['time']
    time_t0 = time[1:]

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
    x_train, y_train, r_train, nan_mask_train = x[train_idx], y[train_idx], ri_t0[train_idx], nan_mask[train_idx]
    x_val, y_val, r_val, nan_mask_val = x[val_idx], y[val_idx], ri_t0[val_idx], nan_mask[val_idx]
    x_test, y_test, r_test, nan_mask_test = x[test_idx], y[test_idx], ri_t0[test_idx], nan_mask[test_idx]

    # Create the destination directory if it doesn't already exist
    os.makedirs(PATH_MODEL_INPUTS, exist_ok = True)

    # Save splits
    np.savez(
        os.path.join(PATH_MODEL_INPUTS, 'train.npz'),
        x_train = x_train,
        y_train = y_train,
        r_train = r_train,
        nan_mask_train = nan_mask_train,
    )

    np.savez(
        os.path.join(PATH_MODEL_INPUTS, 'val.npz'),
        x_val = x_val,
        y_val = y_val,
        r_val = r_val,
        nan_mask_val = nan_mask_val,
    )

    np.savez(
        os.path.join(PATH_MODEL_INPUTS, 'test.npz'),
        x_test = x_test,
        y_test = y_test,
        r_test = r_test,
        nan_mask_test = nan_mask_test,
    )

    print(f'Splits saved at {PATH_MODEL_INPUTS}')

    # Save split indices

    np.savez_compressed(
    os.path.join(PATH_MODEL_INPUTS, f"split_indices.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    
    print(f"Split indices saved at {PATH_MODEL_INPUTS}")

    return


if __name__ == "__main__":
    main()