import numpy as np
import os

from .path import(
    PATH_SOURCE,
    PATH_DEST,
    FSTR_END_IN,
    FSTR_END_OUT,
)

def main():

    # Load in data (same as lr inputs)
    data = np.load(os.path.join(PATH_SOURCE, f'test_{FSTR_END_IN}.npz'))

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
        os.path.join(PATH_DEST, f"preds_ps_{FSTR_END_OUT}"),
        y_pred = y_pred,
        y_true = y_true
    )

    return

if __name__ == "__main__":
    main()

