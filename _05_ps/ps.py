import numpy as np
import os

from _00_config.path import(
    PATH_MODEL_INPUTS,
    PATH_PS_OUT,
)

def main():

    # Load in data (same as lr inputs)
    data = np.load(os.path.join(PATH_MODEL_INPUTS, f'test.npz'))

    # Use test split for persistence baseline
    y_test = data['y_test']

    # Get dimensions for output arrays
    nt, n_out, nlat, nlon = np.shape(y_test)

    # Initialize output arrays
    y_true = np.full((nt - 1, n_out, nlat, nlon), np.nan)
    y_pred = y_true

    # Set predicted equal to previous day (remove last day from array)
    y_pred = y_test[:-1,:,:,:]

    # Shift true array forward
    y_true = y_test[1:,:,:,:]

    # Save predictions
    np.savez(
        os.path.join(PATH_PS_OUT, f"preds_ps.npz"),
        y_pred = y_pred,
        y_true = y_true
    )

    return

if __name__ == "__main__":
    main()

