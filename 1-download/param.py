import os
import sys

def get_vars(var_name: str) -> str:
    """
    Gets global variables from shell script

    Docstring for _get_global_vars
    
    :param var_name: Description
    :type var_name: str
    :return: Description
    :rtype: str
    """

    value = os.getenv(var_name)

    if value is None:
        print(f"ERROR: Set global variable: {var_name}", file = sys.stderror)
        sys.exit(1)

    return value

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Required global variables defined here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = get_vars("HEM") # hemisphere (sh or nh)

START_YEAR = int(get_vars("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(get_vars("END_YEAR")) # data ends 31DEC<END_YEAR>

LAT_LIMITS = [float(x) for x in get_vars("LAT_LIMITS").split(",")] # South to North latitude bounds, degrees
LON_LIMITS = [float(x) for x in get_vars("LON_LIMITS").split(",")] # West to East longitude bounds, degrees

TIMESTAMP_OUT = get_vars("OUT_TIMESTAMP") # Timestamp version for data download

# Nasa Earthdata login credentials for download
USER = os.getenv("USER") # username
PASS = os.getenv("PASS") # password