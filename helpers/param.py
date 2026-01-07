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
        print(f"ERROR: Set global variable: {var_name}", file = sys.stderr)
        sys.exit(1)

    return value

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Required global variables defined here
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Nasa Earthdata login credentials for download
USER = get_vars("USER") # username
PASS = get_vars("PASS") # password