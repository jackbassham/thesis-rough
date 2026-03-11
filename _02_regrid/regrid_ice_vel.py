import numpy as np

from core_regrid import(
    
)

def main(cfg):

    # Load raw data source path
    path_raw = cfg.path_config.data_stage_path('raw')

    # Load regrid data destination path
    path_regrid = cfg.path_config.data_stage_path('regrid')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_regrid)






if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)