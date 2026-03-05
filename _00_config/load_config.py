from .config import DataConfig, VersionConfig, PathConfig


def load_config():

    # Create instance of data parameters
    data_config = DataConfig(
        hemisphere = 'sh',
        year_range = (1992, 2020),
        latitude_bounds = (-80, -62),
        longitude_bounds = (-180, 180),
        grid_resolution = 25
    )

    version_config = VersionConfig()

    paths = PathConfig(data_config, version_config)

    return data_config, version_config, paths