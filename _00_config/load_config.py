from .config import DataConfig, VersionConfig, PathConfig
from .parse_args import parse_args


def load_config():

    # Instantiate argument parser
    args = parse_args()

    # Create instance of data parameters specific to run
    data_config = DataConfig(
        hemisphere = 'sh',
        year_range = (1992, 2020),
        latitude_bounds = (-80, -62),
        longitude_bounds = (-180, 180),
        grid_resolution = 25
    )

    # Create instance of timestamp version
    # With argument parsing for optional data step timestamps,
    # otherwise defaults to current 'timestamp_out'
    version_config = VersionConfig(
        timestamp_out = args.timestamp_out,
        timestamp_raw = args.timestamp_raw,
        timestamp_regrid = args.timestamp_regrid,
        timestamp_coordinates = args.timestamp_coordinates,
        timestamp_mask_norm = args.timestamp_mask_norm,
        timestamp_model_inputs = args.timestamp_model_inputs,
        timestamp_model_output = args.timestamp_model_output,
    )

    # Create instance of paths
    paths = PathConfig(data_config, version_config)

    return data_config, version_config, paths