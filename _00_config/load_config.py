from pathlib import Path
import yaml

from .config import(
    DataConfig,
    DatasetInfo,
    DatasetConfig, 
    VersionConfig, 
    PathConfig, 
    PipelineConfig,
    SplitConfig,
)
from .parse_args import parse_args


def load_config(user_config_path=None):

    if user_config_path is None:
        # Get relative path to user configuration yaml file
        user_config_path = Path(__file__).with_name('user_config.yaml')

    # Load the yaml file
    with open(user_config_path) as file:
        user_config = yaml.safe_load(file)

    # Set the active configuration preset from the yaml file
    preset = user_config['active_preset']

    # Try to get the active preset from presets available in the yaml file
    try:
        p = user_config['presets'][preset]
    # Handle case where active preset is not available in presets
    except KeyError:
        raise ValueError(
            f'Unknown active preset "{preset}".'
            f'Use avaiable presets: {list(user_config["presets"])}'
        )

    # Set data configuration instance from yaml file
    data_config = DataConfig(
        hemisphere = p['hemisphere'],
        year_range = tuple(p['year_range']),
        latitude_bounds = tuple(p['latitude_bounds']),
        longitude_bounds = tuple(p['longitude_bounds']),
        grid_resolution = p['grid_resolution'],
    )

    # Set dataset configuration instance to current name, dataset id, version, grid, and file extension
    dataset_config = DatasetConfig(
        ice_vel=DatasetInfo('ice_vel', 'nsidc0016', 'v4', 'ease', '.npz'),
        wind=DatasetInfo('wind', 'era5', 'v1', 'reg', '.npz'),
        ice_conc=DatasetInfo('ice_conc', 'nsidc0051', 'v2', 'ps', '.npz')
    )

    # Instantiate split configuration object from YAML entries or fall back to defaults
    split_config = SplitConfig(**user_config.get('split', {}))

    # Instantiate argument parser
    args = parse_args()

    # Create instance of timestamp version
    # With argument parsing for optional data step timestamps,
    # otherwise defaults to current timestamp stored in 'timestamp_out'
    version_config = VersionConfig(
        timestamp_out = args.timestamp_out,
        timestamp_raw = args.timestamp_raw,
        timestamp_regrid = args.timestamp_regrid,
        timestamp_coordinates = (
            args.timestamp_coordinates
            or args.timestamp_regrid
        ),
        timestamp_mask_norm = args.timestamp_mask_norm,
        timestamp_model_inputs = args.timestamp_model_inputs,
        timestamp_model_output = args.timestamp_model_output,
    )

    # Instantiate path configuration object to set paths for data, model output, and plots
    path_config = PathConfig(
        data_config, 
        version_config,
        # Get user's data root from yaml file
        # TODO handle user_data_root error
        user_data_root = user_config['paths']['user_data_root']
    )

    # Return entire pipeline configuration object
    return PipelineConfig(
        data_config = data_config,
        dataset_config = dataset_config,
        version_config = version_config,
        path_config = path_config,
    )