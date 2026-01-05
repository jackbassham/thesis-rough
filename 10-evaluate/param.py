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

MODEL_STR = get_vars("MODEL_STR")

# TODO if "TIMESTAMP_MODEL" version given in shell, set to TIMESTAMP_OUT version,
# else set to TIMESTAMP_OUT
TIMESTAMP_OUT = get_vars("TIMESTAMP_OUT")

# If timestamp version not given for model, use TIMESTAMP_OUT (ie: running master shell)
TIMESTAMP_MODEL = os.getenv("TIMESTAMP_MODEL", TIMESTAMP_OUT)

# TODO TIMESTAMP_COORD
# If timestamp version not given for coordinates, use TIMESTAMP_OUT (ie: running master shell)
TIMESTAMP_COORD = get_env("TIMESTAMP_COORD", TIMESTAMP_OUT)

# TODO TIMESTAMP_UNCERTAINTY
# If timestamp version not given for uncertainty, use TIMESTAMP_OUT (ie: running master shell)
# Uncertainty taken from LR test split
TIMESTAMP_R = get_env("TIMESTAMP_R", TIMESTAMP_OUT)
