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
    data_config, version_config, paths = load_config()


def load_config():
    """
    --------------------- NOTE ----------------------------
    temporal bounds for current reproduction
    year_range = (1992, 2020) (NOTE 1989-1991 JRA55 issue)
    latitude longitude bounds for current reproduction
    north: latitude_bounds = (60, 90)        
    south: latitdue_bounds = (-80, -62)
    all: longitude_bounds = (-180, 180)
    -------------------------------------------------------
    """

    # Create instance of data parameters specific to run
    # NOTE TEST INSTANCE, Weddell Sea,  3 years
    data_config = DataConfig(
        hemisphere = 'south',
        year_range = (1992, 1995),
        latitude_bounds = (-79, -60),
        longitude_bounds = (-70, -15),
        grid_resolution = 25
    )

    # # Create instance of data parameters specific to run
    # data_config = DataConfig(
    #     hemisphere = 'south',
    #     year_range = (1992, 2020),
    #     latitude_bounds = (-80, -62),
    #     longitude_bounds = (-180, 180),
    #     grid_resolution = 25
    # )

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
    path_config = PathConfig(data_config, version_config)

    # Return entire pipeline configuration object
    return PipelineConfig(
        data_config = data_config,
        dataset_config = dataset_config,
        version_config = version_config,
        path_config = path_config,
    )


if __name__ == '__main__':
    main()