from dataclasses import dataclass

class Config:
    def __init__(self,
            hemisphere, year_range,
            latitude_bounds, longitude_bounds, grid_resolution,
):
        """
        Configuration for global data parameters

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
        self.hempisphere = hemisphere
        self.year_range = year_range
        self.latitude_bounds = latitude_bounds
        self.longitude_bounds = longitude_bounds
        self.grid_resolution = grid_resolution

        # Getter for hemisphere
        @property
        def hemisphere(self):
            return self._hemisphere
        
        # Setter for hemisphere
        @hemisphere.setter
        def hemisphere(self, hemisphere):
            # Handle case where hemisphere string is invalid
            if not hemisphere == 'nh' or not hemisphere == 'sh':
                raise ValueError('Invalid hemisphere string: Enter "sh" for Southern or "nh" for Northern') 
            self._hemisphere = hemisphere

        @property
        def year_range(self):
            return self._year_range
        
        @year_range.setter
        def year_range(self, year_range):
            # Handle case where years are not entered as a tuple range
            if not isinstance(year_range, tuple) or not len(year_range) == 2:
                raise ValueError('Enter a tuple year range (min year, max year)')
            
            # Extract start year and end year
            start, end = year_range

            # Handle case where either year is not valid
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError('Invalid year input: Enter years in format YYYY')

            # Handle case where years are out of range
            if not 1989 < year_range < 2020:
                raise ValueError('Years out of range: Enter in range 1989 to 2020')


# Create instance of parameters for model run
config = Config(
    hempisphere = 'sh',
    year_range = (1992, 2020),
    latitude_limts = (-80, -62),
    longitude_limits = (-180, 180),
    grid_resolution = 25,
)
    

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

