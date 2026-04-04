import numpy as np
import numpy.typing as npt
from pathlib import Path
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path


def load_npz_data(file_path: Path) -> dict[str, npt.NDArray]:
    # Try to load the raw data
    try:
        # Get data dict {variable_name: array} from numpy data loader
        return dict(np.load(file_path))

    # Handle case where file does not exist
    except FileNotFoundError:
        sys.exit(f'File at {file_path} not found')


def load_ice_vel(path: Path, filename: str):
    """
    
    """

    # Load the data
    data = load_npz_data(path / filename)

    # Extract variables
    ui = data['ui']
    vi = data['vi']
    ri = data['ri']
    
    # Delete data dict
    del data

    return ui, vi, ri


def load_wind(path: Path, filename: str):
    """
    
    """

    # Load the data
    data = load_npz_data(path / filename)

    # Extract variables
    ua = data['ua']
    va = data['va']

    # Delete data dict
    del data

    return ua, va


def load_ice_conc(path: Path, filename: str):
    """
    
    """

    # Load the data
    data = load_npz_data(path / filename)

    # Extract variables
    ci = data['ci']

    # Delete data dict
    del data

    return ci