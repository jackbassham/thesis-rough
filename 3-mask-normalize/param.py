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

TIMESTAMP_IN = get_vars("TIMESTAMP_IN") # timestamp version for regrid input data
TIMESTAMP_OUT = get_vars("TIMESTAMP_OUT") # timestamp version for masked, normalized data
