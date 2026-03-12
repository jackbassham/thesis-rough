import numpy as np
import sys

def load_npz_data(file_path):
    # Try to load the raw data
    try:
        # Get data dict {variable_name: array} from numpy data loader
        return dict(np.load(file_path))

    # Handle case where file does not exist
    except FileNotFoundError:
        sys.exit(f'File at {file_path} not found')