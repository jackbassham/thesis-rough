import numpy as np
import numpy.typing as npt
from pathlib import Path
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path

def load_npz_data(file_path: Path) -> dict[str, npt.NDarray]:
    # Try to load the raw data
    try:
        # Get data dict {variable_name: array} from numpy data loader
        return dict(np.load(file_path))

    # Handle case where file does not exist
    except FileNotFoundError:
        sys.exit(f'File at {file_path} not found')