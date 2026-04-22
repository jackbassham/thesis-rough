from .config import(
    DataConfig,
    DatasetInfo,
    DatasetConfig, 
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
    """
    --------------------- NOTE ----------------------------
    temporal bounds for full-region reproduction
    year_range = (1992, 2020) (NOTE 1989-1991 JRA55 issue)
    latitude longitude bounds for current reproduction
    north: latitude_bounds = (60, 90)        
    south: latitdue_bounds = (-80, -62)
    all: longitude_bounds = (-180, 180)
    -------------------------------------------------------
    """

    # FIXME move data root and data config to .yaml file

    # Define root to data directory
    # If 'USER_DATA_ROOT = None', data is downloaded to repo directory
    USER_DATA_ROOT = None
    # USER_DATA_ROOT = '/data/globus/jbassham/thesis-rough'

    # Create instance of data parameters specific to run
    data_config = DataConfig(
        hemisphere = 'south',
        year_range = (1992, 2020), # At least 6 years
        latitude_bounds = (-80, -62),
        longitude_bounds = (-180, 180),
        grid_resolution = 25
    )

    # FIXME use .netrc (Earthdata rec) for login credentials, or prompt user
    # Create instance of login credentials for Nasa Earth Data access
    Earthdata_login_credentials = LoginCredentials(
        username = 'jbassham',
        password = '$EarthDataPass2026'
    )

    # Create configuration instance of dataset info
    dataset_config = DatasetConfig(
        ice_vel=DatasetInfo('ice_vel', 'nsidc0016', 'v4', 'ease', '.npz'),
        wind=DatasetInfo('wind', 'era5', 'v1', 'reg', '.npz'),
        ice_conc=DatasetInfo('ice_conc', 'nsidc0051', 'v2', 'ps', '.npz')
    )

    # Instantiate argument parser
    args = parse_args()

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
    path_config = PathConfig(data_config, version_config, user_data_root = USER_DATA_ROOT)

    # Return entire pipeline configuration object
    return PipelineConfig(
        data_config = data_config,
        dataset_config = dataset_config,
        version_config = version_config,
        path_config = path_config,
        login_credentials = Earthdata_login_credentials,
    )


if __name__ == '__main__':
    main()