from dataclasses import dataclass
from datetime import datetime
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
        'sh' Southern Hemisphere
        'nh' Northern Hemisphere

        Temporal Bounds:
        year_range = (1992, 2020)
        # NOTE: Original Hoffman, et. al range is 1989-2020, 1989-1991 corrupted JRA55 data files on Mazloff Server
        # TODO: Script to JRA55 or ERA data from source

        Spatial Bounds: 
        Southern Hemisphere; Southern Ocean
        latitude_limits = (-80, -62), limited to -90degS to -37degS
        longitude_limts = (-180, 180), limited to -180degW to 180degE
        
        Northern Hemisphere; Arctic ('nh')
        latitude_limits = (60, 90), limited to 29.7N to 90N
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
        # Handle invalid hemisphere string
        if self.hemisphere not in ('sh', 'nh'):
            raise ValueError('Invalid hemisphere string: Enter "sh" for Southern or "nh" for Northern') 
        

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
        if start < 1989 or end > 2020:
            raise ValueError('Years out of range: Enter in range 1989 to 2020')

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
        if self.hemisphere == 'sh':
            if min_lat < -90 or max_lat > -37:
                raise ValueError('Southern Hemisphere latitude bounds limited to (-90, -37) ie: "90degS, 37degS"')

        # Handle invalid latitude bounds for Northern Hemisphere
        if self.hemisphere == 'nh':
            if min_lat < 29.7 or max_lat > 90:
                raise ValueError('Northern Hemisphere latitude bounds limited to (29.7, 90) ie: "29.7degN, 90degN"') 
        

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
class VersionConfig:
    """
    Configuration for timestamped data version control.

    Assigns one timestamp at runtime to 'timestamp_out', unless
    user assigns 'timestamp_out' version manually when instantiating the version 
    configuration. Timestamps can be optionally assigned for different versions of data to allow
    user to run from different start points. These default to 'timestamp_out' if not assigned by user 
    (ie: user runs the entire pipeline)
    """

    # Define format for timestamp version
    _timestamp_format = "%m%d%Y_%H%M"

    timestamp_out: Optional[str] = None

    timestamp_raw: Optional[str] = None
    timestamp_regrid: Optional[str] = None
    timestamp_mask_norm: Optional[str] = None
    timestamp_inputs: Optional[str] = None
    timestamp_outputs: Optional[str] = None
    timestamp_coordinates: Optional[str] = None
    timestamp_nan_mask: Optional[str] = None
    timestamp_uncertainty: Optional[str] = None

    @classmethod
    def get_timestamp(cls):

        # Generate time stamp with format #MMDDYY_HHMM
        return(datetime.now().strftime(cls._timestamp_format))


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

        # Iterate through remaining timestamps
        for attr in [
            'timestamp_raw',
            'timestamp_regrid',
            'timestamp_mask_norm',
            'timestamp_inputs',
            'timestamp_outputs',
            'timestamp_coordinates',
            'timestamp_nan_mask',
            'timestamp_uncertainty',
        ]:
            
            # Get the input value of timestamp attribute
            value = getattr(self, attr)

            # If user does not enter manually  
            if value is None:
                # Set to current timestamp (default)
                setattr(self, attr, self.timestamp_out)

            else:
                # Check for errors in manually entered timestamp
                self.validate_format(value)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter Validation Methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _validate_format(self, timestamp: str):
        try:
            datetime.strptime(timestamp, self._timestamp_format)

        except ValueError:
            raise ValueError(f'Timestamp {timestamp} must match format {self._timestamp_format}')
        

@dataclass
class LoginCredentials:
    """
    Login credentials 
    """

    username: str
    password: str


def main():

    # Create instance of parameters for model run
    dataconfig = DataConfig(
        hemisphere = 'sh',
        year_range = (1992, 2020),
        latitude_bounds = (-80, -62),
        longitude_bounds = (-180, 180),
        grid_resolution = 25
    )


if __name__ == '__main__':
    main()
    

def get_global_variable(shell_variable: str, default_variable: str | None = None) -> str:
    """
    NOTE including type hints above for string input ': str' and output '-> str'
    
    Returns global variable name set by user in shell script as a string

    If user does not set variable in shell script, variable defaults to None 
    and the program terminates with an error message.

    Default variable may be set to some other default variable at function asignment
    (ie: a specific timestamp version).
    """

    # Get variable from shell script
    global_variable = os.getenv(shell_variable, default_variable)

    # Handle case where variable is missing from shell script
    if global_variable is None:
        raise RuntimeError(f'Set variable "{shell_variable}" in the shell script')

    return global_variable


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Required global variables retrieved here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = get_global_variable("HEM") # hemisphere (sh or nh)
START_YEAR = int(get_global_variable("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(get_global_variable("END_YEAR")) # data ends 31DEC<END_YEAR>

LAT_LIMITS = [float(x) for x in get_global_variable("LAT_LIMITS").split(",")] # South to North latitude bounds, degrees
LON_LIMITS = [float(x) for x in get_global_variable("LON_LIMITS").split(",")] # West to East longitude bounds, degrees

RESOLUTION = int(get_global_variable("RESOLUTION")) # Grid resolution, km

# Nasa Earthdata login credentials for download
USER = get_global_variable("USER") # username
PASS = get_global_variable("PASS") # password

# Model name string identifier
MODEL_STR = get_global_variable("MODEL_STR")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Timestamps or timestamp defaults retrieved here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TIMESTAMP_OUT = get_global_variable("TIMESTAMP_OUT")
TIMESTAMP_RAW = get_global_variable("TIMESTAMP_RAW", TIMESTAMP_OUT)
TIMESTAMP_REGRID = get_global_variable("TIMESTAMP_REGRID", TIMESTAMP_OUT)
TIMESTAMP_COORDINATES = get_global_variable("TIMESTAMP_COORDINATES", TIMESTAMP_OUT)
TIMESTAMP_NAN_MASK = get_global_variable("TIMESTAMP_NAN_MASK", TIMESTAMP_OUT)
TIMESTAMP_R = get_global_variable("TIMESTAMP_UNCERTAINTY", TIMESTAMP_OUT)
TIMESTAMP_MASK_NORM = get_global_variable("TIMESTAMP_MASK_NORM", TIMESTAMP_OUT)
TIMESTAMP_INPUTS = get_global_variable("TIMESTAMP_INPUTS", TIMESTAMP_OUT)
TIMESTAMP_OUTPUTS = get_global_variable("TIMESTAMP_OUTPUTS", TIMESTAMP_OUT)

