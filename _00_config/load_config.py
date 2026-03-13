from .config import(
    DataConfig, 
    VersionConfig, 
    PathConfig, 
    LoginCredentials,
    PipelineConfig,
)
from .parse_args import parse_args


def main():

    # Load in configuration defined in function below
    data_config, version_config, paths, login_credentials = load_config()


def load_config():

    # Instantiate argument parser
    args = parse_args()

    # Create instance of data parameters specific to run
    data_config = DataConfig(
        hemisphere = 'south',
        year_range = (1992, 2020),
        latitude_bounds = (-80, -62),
        longitude_bounds = (-180, 180),
        grid_resolution = 25
    )

    # Create instance of timestamp version
    # With argument parsing for optional data step timestamps,
    # otherwise defaults to current timestamp stored in 'timestamp_out'
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
    path_config = PathConfig(data_config, version_config)

    # Create instance of login credentials for Nasa Earth Data access
    login_credentials = LoginCredentials(
        username = 'jbassham',
        password = 'guJdib-huczi6-jimsuh'
    )

    # Return entire pipeline configuration object
    return PipelineConfig(
        data_config = data_config,
        version_config = version_config,
        path_config = path_config,
        login_credentials = login_credentials,
    )


if __name__ == '__main__':
    main()