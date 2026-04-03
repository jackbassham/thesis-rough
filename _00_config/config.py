from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# TODO consider YAML

@dataclass
class DataConfig:
    """
    Configuration for global data data parameters

    Parameters used for replication:

        Enter parameters based on the following for replication of Northern Hemisphere 
        and Southern Hemisphere model runs.

        Hemisphere String identifier:
        String identifier
        'south' Southern Hemisphere
        'north' Northern Hemisphere

        Temporal Bounds:
        year_range = (1992, 2020)
        # NOTE: Original Hoffman, et. al range is 1989-2020, 1989-1991 corrupted JRA55 data files on Mazloff Server
        # TODO: Script to JRA55 or ERA data from source

        Spatial Bounds for Reproduction: 
        Southern Hemisphere; Southern Ocean
        latitude_limits = (-80, -62), limited to -90degS to -37degS
        (max latitude bounds based on datasets (-90, -40))
        longitude_limts = (-180, 180), limited to -180degW to 180degE
        
        Northern Hemisphere; Arctic ('nh')
        latitude_limits = (60, 90), limited to 29.7N to 90N
        (max latitude bounds based on datasets (31, 90))
        longitude_limts = (-180, 180), limited to -180degW to 180degE

        Grid Resolution:
        25
        Resolution, in km, used for new regrid data. Based on original EASE grid
        resolution of NSIDC Polar Pathfinder Sea Ice Motion Vectors, version 4. 
        Converted to degrees during regrid. 

    """

    # Construct parameters
    hemisphere: str
    year_range: Tuple[int, int]
    latitude_bounds: Tuple[float, float]
    longitude_bounds: Tuple[float, float]
    grid_resolution: int


    def __post_init__(self):
        """
        Post parameter initialization error handling using validation methods
        """ 
        self._validate_hemisphere()
        self._validate_year_range()
        self._validate_latitude_bounds()
        self._validate_longitude_bounds()
        self._validate_grid_resolution()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter Validation Methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _validate_hemisphere(self):
        # Allow for case insensitivity and whitespace
        self.hemisphere = self.hemisphere.lower().strip()

        # Handle invalid hemisphere string
        if self.hemisphere not in ('south', 'north'):
            raise ValueError('Invalid hemisphere string: Enter "south" or "north"') 
        

    def _validate_year_range(self):
        # Handle years not entered as range
        if not len(self.year_range) == 2:
            raise ValueError('Enter a year range as a tuple (start year, end year)')
            
        # Extract start year and end year from tuple
        start, end = self.year_range

        # Handle invalid year entry type
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError('Invalid year input: Enter years as integers in format YYYY')

        # Handle years out of range
        if start < 1989 or end > 2024:
            raise ValueError('Years out of range: Enter in range 1989 to 2024')

        # Handle invalid order of range
        if not start < end:
            raise ValueError('Start year must precede end year')


    def _validate_latitude_bounds(self):
        # Handle latitude bounds not entered as range
        if not len(self.latitude_bounds) == 2:
            raise ValueError('Enter a latitude bounds as a tuple (min lat, max lat)')
        
        # Extract minimum and maximum latitudes
        min_lat, max_lat = self.latitude_bounds

        # Handle invalid latitude entry type
        if not isinstance(min_lat, (float, int)) or not isinstance(max_lat, (float, int)):
            raise ValueError('Invalid latitude bound input: Enter latitudes as integers or floats')
        
        # Handle miss-ordered latitude bounds
        if not min_lat < max_lat:
            raise ValueError('Enter latitude bounds from South to North (where minimum less than maximum)')

        # Handle invalid latitude bounds for Southern Hemisphere
        if self.hemisphere.lower().strip() == 'south':
            if min_lat < -90 or max_lat > -40:
                raise ValueError('Southern Hemisphere latitude bounds limited to (-90, -40) ie: "90degS, 40degS"')

        # Handle invalid latitude bounds for Northern Hemisphere
        if self.hemisphere.lower().strip() == 'north':
            if min_lat < 31 or max_lat > 90:
                raise ValueError('Northern Hemisphere latitude bounds limited to (31, 90) ie: "31degN, 90degN"') 
        

    def _validate_longitude_bounds(self):
        # Handle longitude bounds not entered as range
        if not len(self.longitude_bounds) == 2:
            raise ValueError('Enter latitude as a tuple (min lat, max lat)')
        
        # Extract minimum and maximum longitude from bounds
        min_lon, max_lon = self.longitude_bounds

        # Handle invalid latitude entry type
        if not isinstance(min_lon, (float, int)) or not isinstance(max_lon, (float, int)):
            raise ValueError('Longitude bounds must be entered as integers or floats')
        
        # Handle miss-ordered longitude bounds
        if not min_lon < max_lon:
            raise ValueError('Enter longitude bounds from West to East (-180 to 180; where minimum less than maximum)')

        # Handle longitude bounds out of range
        if min_lon < -180 or max_lon > 180:
            raise ValueError('Longitude bounds limited to (-180, 180) ie: "180degW, 180degE"')
        

    def _validate_grid_resolution(self):
        # Handle invalid grid resolution
        if not isinstance(self.grid_resolution, (float, int)) or self.grid_resolution <= 0:
            raise ValueError('Invalid grid resolution, enter as positive, nonzero integer or float')
        

@dataclass
class DatasetInfo:
    """

    """

    # Construct parameters
    name: str
    id: str
    version: str
    original_grid: str
    ext: str



@dataclass
class DatasetConfig:
    """
    
    """

    # Construct parameters
    ice_vel: DatasetInfo
    wind: DatasetInfo
    ice_conc: DatasetInfo

    # List valid dataset states
    DATASET_STAGES = [
        'raw', 
        'regrid',
    ]

    # Construct dict of datasets
    def datasets(self):
        """
        
        """
        return{
            'ice_vel': self.ice_vel,
            'wind': self.wind,
            'ice_conc': self.ice_conc,
        }

    def build_filename(self, ds: DatasetInfo, stage: str) -> str:
        """
        
        """
        
        # Handle case where stage not properly defined in script
        if stage not in self.DATASET_STAGES:
            raise ValueError(f'Unknown dataset stage: {stage}')
        
        # Construct base of filename
        base = f'{ds.name}_{stage}_{ds.id}_{ds.version}'
        
        # Handle raw data case
        if stage == 'raw':
            # Include original grid in filename
            base += f'_{ds.original_grid}'
        
        # Return full filename with extension
        return base + ds.ext
        

@dataclass
class VersionConfig:
    """
    Configuration for timestamped data version control.

    Assigns one timestamp at runtime to 'timestamp_out', unless
    user assigns 'timestamp_out' version manually when instantiating the version 
    configuration. Timestamps can be optionally assigned for version of specific data stage to allow
    user to run from different start points. These default to 'timestamp_out' if not assigned by user 
    (ie: user runs the entire pipeline)
    """

    # Define format for timestamp version
    _timestamp_format = "%m%d%Y_%H%M"

    timestamp_out: Optional[str] = None

    timestamp_raw: Optional[str] = None
    timestamp_regrid: Optional[str] = None
    timestamp_coordinates: Optional[str] = None
    timestamp_mask_norm: Optional[str] = None
    timestamp_model_inputs: Optional[str] = None
    timestamp_model_output: Optional[str] = None


    def get_timestamp(self):

        # Generate time stamp with format #MMDDYY_HHMM
        return(datetime.now().strftime(self._timestamp_format))


    def __post_init__(self):
        """
        Post parameter initialization error handling using validation methods. 
        Sets timestamps to current 'timestamp_out', unless user specifies manually.
        Raise value error if user does not enter in format MMDDYY_HHMM.
        """ 

        # If user does not enter manually, assign current timestamp
        if self.timestamp_out is None:
            self.timestamp_out = self.get_timestamp()
        else:
            # Check for errors in manually entered timestamp
            self._validate_format(self.timestamp_out)

        # Iterate through remaining data stage timestamps
        for attr in [
            'timestamp_raw',
            'timestamp_regrid',
            'timestamp_coordinates',
            'timestamp_mask_norm',
            'timestamp_model_inputs',
            'timestamp_model_output',
        ]:
            
            # Get the input value of timestamp attribute
            value = getattr(self, attr)

            # If user does not enter manually  
            if value is None:
                # Set to current timestamp (default)
                setattr(self, attr, self.timestamp_out)

            else:
                # Check for errors in manually entered timestamp
                self._validate_format(value)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter Validation Methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _validate_format(self, timestamp: str):
        try:
            datetime.strptime(timestamp, self._timestamp_format)

        except ValueError:
            raise ValueError(f'Timestamp {timestamp} must match format {self._timestamp_format}')
        

class PathConfig:
    """
    
    """

    # TODO: __str__() method to print for paths in main function here 

    # List valid data stages
    DATA_STAGES = [
        'raw',
        'regrid',
        'coordinates',
        'mask_norm',
        'model_inputs',
        'model_output',
    ]

    # List valid model names
    MODEL_NAMES = [
        'ps',
        'lr-cf',
        'lr-cf-wtd',
        'cnn-pt',
        'cnn-pt-wtd',
    ]

    # Pass in instance of data configuratino and version configuration
    def __init__(self, data_config: DataConfig, version_config: VersionConfig):

        # Instantiate configuration objects
        self.data_config = data_config
        self.version_config = version_config

        # Define root to Mazloff scratch data directory
        # TODO: allow dynamic/ change for other users
        # when working with small sample dataset
        self.data_root = Path('/data/globus/jbassham/thesis-rough')

        # Define root to project directory for plots, etc
        self.project_root = Path('.')


    def data_stage_path(self, data_stage: str) -> Path:
        """
        Instance method for creating path to specific data stage
        """

        # Handle case where data stage entry is invalid
        if data_stage not in self.DATA_STAGES:
            raise ValueError(f'Unknown data stage entry in path config: {data_stage}')

        # Get timestamp attribute for data stage
        timestamp = getattr(self.version_config, f'timestamp_{data_stage}')

        # Return path for the data stage
        return Path(self.data_root / data_stage / self.data_config.hemisphere / timestamp)
    

    def model_path(self, model_name: str, plot_path: bool = False) -> Path:
        """
        Instance method for creating path to specific model outputs
        If plot_path is set to True, creates path to quick eval plots in project directory
        """

        # Handle case where model name entry is invalid
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f'Uknown model name entry in path config: {model_name}')

        # Get timestamp for model output
        timestamp = self.version_config.timestamp_model_outputs

        # Return path for the quick evaluation plots
        if plot_path:
            return self.project_root / 'plots' / 'quick-eval' / model_name / self.data_config.hemisphere / timestamp
        
        # Return path for the model outputs
        else:
            return self.data_root / 'model-output' / model_name / self.data_config.hemisphere / timestamp


    def makedir_if_missing(self, path: Path) -> Path:
        """
        Instance method makes directory if it doesn't already exist.
        parents = True and exist_ok = True prevents path.mkdir from
        raising FileExistsError.
        """
        path.mkdir(parents=True, exist_ok = True)
        return path
        

@dataclass
class LoginCredentials:
    """
    Login credentials for sites where account is required for data download
    ie: NSIDC requires Nasa Earth Data Login
    """

    username: str
    password: str


@dataclass
class PipelineConfig:
    """
    Container for all pipeline configuration objects.
    """
    data_config: DataConfig
    dataset_config: DatasetConfig
    version_config: VersionConfig
    path_config: PathConfig
    login_credentials: LoginCredentials


def main():

    # Create instance of data parameters
    dataconfig = DataConfig(
        hemisphere = 'south',
        year_range = (1992, 2020),
        latitude_bounds = (-80, -62),
        longitude_bounds = (-180, 180),
        grid_resolution = 25
    )

    # Create instance timestamp version
    versionconfig = VersionConfig()

    # Create instance of login credentials
    logincredentials = LoginCredentials(
        username = 'jbassham', 
        password = 'guJdib-huczi6-jimsuh'
    )


if __name__ == '__main__':
    main()

