import os


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
TIMESTAMP_COORDINATES = get_global_variable("TIMSTAMP_COORDINATES", TIMESTAMP_OUT)
TIMESTAMP_NAN_MASK = get_global_variable("TIMESTAMP_NAN_MASK", TIMESTAMP_OUT)
TIMESTAMP_R = get_global_variable("TIMESTAMP_UNCERTAINTY", TIMESTAMP_OUT)
TIMESTAMP_MASK_NORM = get_global_variable("TIMESTAMP_MASK_NORM", TIMESTAMP_OUT)
TIMESTAMP_INPUTS = get_global_variable("TIMESTAMP_INPUTS", TIMESTAMP_OUT)
TIMESTAMP_OUTPUTS = get_global_variable("TIMESTAMP_OUTPUTS", TIMESTAMP_OUT)

