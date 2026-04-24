from .config import(
    DataConfig,
    DatasetInfo,
    DatasetConfig, 
    VersionConfig, 
    PathConfig, 
    PipelineConfig,
)
from .parse_args import parse_args


def load_config():

    # Define root to data directory
    # If 'USER_DATA_ROOT = None', data is downloaded to repo directory
    USER_DATA_ROOT = None
    # USER_DATA_ROOT = '/data/globus/jbassham/thesis-rough'

    # Weddell Sea Test Set
    data_config = DataConfig(
        hemisphere = 'south',
        year_range = (2010, 2016), # At least 6 years
        latitude_bounds = (-79, -62), # Weddell Sea, small subset
        longitude_bounds = (-70, -15),
        grid_resolution = 25,
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
        timestamp_coordinates = (
            args.timestamp_coordinates
            or args.timestamp_regrid
        ),
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
    )


if __name__ == '__main__':
    main()