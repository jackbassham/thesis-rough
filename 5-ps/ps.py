import numpy as np
import os

# TODO import paths
from .path import(

)


def main():

    # Load in data (same as lr inputs)
    data = np.load(os.path.join(PATH_SOURCE, f'test_{FSTR_END_IN}'))

    # Use test split for persistence baseline
    y_test = data['y_test']

    # Get dimensions for output arrays
    nt, n_out, nlat, nlon = np.shape(y_test)

    # Initialize output arrays
    y_true = np.full((nt - 1, n_out, nlat, nlon))
    y_pred = true_all

    # Set predicted equal to previous day (remove last day from array)
    y_pred = y_test[:-1,:,:,:]

    # Shift true array forward
    y_true = y_test[1:,:,:,:]

    # TODO Save arrays

    return

    if __name__ == "__main__":
        main()

