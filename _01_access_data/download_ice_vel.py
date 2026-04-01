import numpy as np
from pathlib import Path
import xarray as xr
from typing import Tuple, Generator, TYPE_CHECKING
if TYPE_CHECKING:
    from _00_config.config import PipelineConfig

def main(cfg: PipelineConfig):

    # Load raw data destination path
    path_raw = cfg.path_config.data_stage_path('raw')
    
    # Define raw data destination file name
    filename = 'ice_vel_raw_ppv4_ease.npz'

    ...





    
if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)