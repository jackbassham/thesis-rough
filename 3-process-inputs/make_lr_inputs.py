import gc
import torch
import numpy as np
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master '<  >.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

TIMESTAMP_IN = os.getenv("TIMESTAMP_IN") # timestamp version of input data

TIMESTAMP_COORD = os.getenv("TIMESTAMP_COORD") # timestamp version of coordinate data

TIMESTAMP_OUT = os.getenv("TIMESTAMP_OUT") # timestamp version of inputs processed here

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define masked & normalized input data path; relative to current
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        'mask-norm', 
        HEM,
        TIMESTAMP_IN)
)

# Define path to coordinate variables
PATH_COORD = os.path.abspath(
    os.path.join(
        script_dir,
        '..',
        'data',
        'coordinates',
        HEM,
        TIMESTAMP_COORD
    )
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_SOURCE, exist_ok=True)

# Define model output data input path; relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        'lr-input',
        HEM,
        TIMESTAMP_OUT)
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additonal global variables here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}"
FSTR_END_COORD = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_COORD}"
FSTR_END_OUT = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_OUT}"


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

    # Get dimensions
    nt, nlat, nlon = np.shape(ui) # time, latitude, longitude

    # Convert NaN values in uncertainty to 1000 (flag)
    ri_filt = np.where(np.isnan(ri), 1e3, ri)

    print("Uncertainty NaNs Converted to 1000")

    # Delete arrays to free memory
    del ui, vi, ua, va, ci, ri
    gc.collect() 

    # Extract time (dates)
    fnam = f"coordinates_{FSTR_END_COORD}.npz"
    data = np.load(os.path.join(PATH_COORD, fnam), allow_pickle=True)
    time = data['time']

    # Create present day parameters (t0) by shifting forward one day
    ui_t0 = ui[1:,:,:]
    vi_t0 = vi[1:,:,:]
    ua_t0 = ua[1:,:,:]
    va_t0 = va[1:,:,:]
    ri_t0 = ri_filt[1:,:,:]

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
    ri_t0 = ri_t0.unsqueeze(1) # [nt, 1, nlat, nlon]

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

    np.savez(
        os.path.join(PATH_DEST, f'train_{FSTR_END_OUT}.npz'),
        x_train = x_train,
        y_train = y_train,
        r_train = r_train
    )

    np.savez(
        os.path.join(PATH_DEST, f'val_{FSTR_END_OUT}.npz'),
        x_val = x_val,
        y_val = y_val,
        r_val = r_val
    )

    np.savez(
        os.path.join(PATH_DEST, f'test_{FSTR_END_OUT}.npz'),
        x_test = x_test,
        y_test = y_test,
        r_test = r_test,
    )

    print(f"Train, Validation, and Test splits saved at {PATH_DEST}")

    # Save split indices

    np.savez_compressed(
    os.path.join(PATH_DEST, f"split_indices_lr_{FSTR_END_OUT}.npz"),
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
    )
    
    print(f"Split indices saved at {PATH_DEST}")

    return

if __name__ == "__main__":
    main()