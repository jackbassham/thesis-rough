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

# Create the directory if it doesn't already exist
os.makedirs(PATH_SOURCE, exist_ok=True)

# Define model output data input path; relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        'cnn-input',
        HEM,
        TIMESTAMP_OUT)
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)

# Set random seed for reproducibility

def set_seed(seed=42):
    torch.manual_seed(seed) # PyTorch Reproducibility
    torch.cuda.manual_seed(seed) # Required if using GPU
    torch.backends.cudnn.deterministic = True  # Reproducibility if using GPU
    torch.backends.cudnn.benchmark = False # Paired with above

    return

def main():

    # Get current script directory path
    script_dir = os.path.dirname(__file__)

    # Navigate to data source directory from current path
    PATH_SOURCE = os.path.join(script_dir, '..', '..', 'data', HEM, 'masked-normalized')

    # Get absolute path to data source directory
    PATH_SOURCE = os.path.abspath(PATH_SOURCE)

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

    print("Input Variables Loaded")

    # Get landmask (where always nan) (USE LATER FOR PLOTTING!)
    land_mask = np.all(np.isnan(uit), axis = 0)

    # Get dimensions
    nt, ny, nx = np.shape(uit) # time, latitude, longitude

    # Convert NaN values in inputs to zero
    uit_filt = np.nan_to_num(uit, 0)
    vit_filt = np.nan_to_num(vit, 0)
    uwt_filt = np.nan_to_num(uwt, 0)
    vwt_filt = np.nan_to_num(vwt, 0)
    icy_filt = np.nan_to_num(icy, 0)

    print("Input NaNs Converted to 0")

    # Convert NaN values in uncertainty to 1000 (flag)
    rt_filt = np.where(np.isnan(rt), 1e3, rt)

    print("Uncertainty NaNs Converted to 1000")

    # Delete arrays to free memory
    del uit, vit, uwt, vwt, icy, rt
    gc.collect() 

    # Extract time (dates)
    fnam = f"time_today_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    data = np.load(os.path.join(PATH_SOURCE, fnam), allow_pickle=True)
    tt = data['time_today']

    print("Time Loaded")

    # Define number of input channels
    n_in = 3

    # Define number of output channels
    n_out = 2

    # Initialize PyTorch Tensors (batch, channels, height, width)
    x = torch.zeros((nt, n_in, ny, nx), dtype = torch.float32) # Features
    y = torch.zeros((nt, n_out, ny, nx), dtype = torch.float32) # Targets

    # Fill feature arrays
    x[:, 0, :, :] = torch.from_numpy(uwt_filt) # Zonal Wind, Today
    x[:, 1, :, :] = torch.from_numpy(vwt_filt) # Meridional Wind, Today
    x[:, 2, :, :] = torch.from_numpy(icy_filt) # Ice Concentration, Yesterday

    # Fill target arrays
    y[:, 0, :, :] = torch.from_numpy(uit_filt) # Zonal Ice Velocity, Today
    y[:, 1, :, :] = torch.from_numpy(vit_filt) # Meridional Ice Velocity, Today

    print("Feature and Target Arrays filled")

    # Reshape uncertainty
    r = torch.from_numpy(rt_filt)
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

    # Create destination path for inputs, if it doesn't exist
    PATH_DEST = os.path.join(script_dir, '..', '..', 'data', HEM, 'cnn-inputs')
    
    # Get absolute path to data destination directory
    PATH_DEST = os.path.abspath(PATH_DEST)
    os.makedirs(PATH_DEST, exist_ok=True)

    end_str = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP}"

    torch.save(
        (x_train, y_train, r_train), 
        os.path.join(PATH_DEST, f'train_{end_str}.pt')
        )
    torch.save(
        (x_val, y_val, r_val), 
        os.path.join(PATH_DEST, 
        f'val_{end_str}.pt')
        )
    torch.save((x_test, y_test, r_test), 
        os.path.join(PATH_DEST, 
        f'test_{end_str}.pt')
        )

    print(f"Train, Validation, and Test splits saved at {PATH_DEST}")

    np.savez_compressed(
    os.path.join(PATH_DEST, f"indices_cnn{HEM}{START_YEAR}{END_YEAR}.npz"),
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    land_mask = land_mask
)
    
    print(f"Train, Validation, and Test indices saved at {PATH_DEST}")

    return

if __name__ == "__main__":
    main()




