import requests

from _00_config.load_config import load_config


"""
NOTE: 
Nasa Earth Data authorisation and access code adapted from:
1. Earth Data Login Documentation, 'How to Access Data with Python', https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python
2. NSIDC example script; NSIDC_Parse_HTML_BatchDL.py
with help from ChatGPT
"""

# Define Nasa Earthdata Host
EARTHDATA_HOST = 'urs.earthdata.nasa.gov'


class EarthdataSession(requests.Session):
    """
    Adapted from authentication SessionWithHeaderRedirection'. Create class that inherits from 
    the class requests.Sesssion (extending class provided by Requests) with rebuild_auth method change
    because of redirects with NASA Earthdata's authentication system.
    """

    # Define host where authentication headers maintained
    global EARTHDATA_HOST
    AUTH_HOST = EARTHDATA_HOST

    # FIXME move username and password from LoginCredentials to .netrc file
    # See https://nsidc.org/data/user-resources/help-center/creating-netrc-file-earthdata-login
    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        """
        Overrides from the library to keep headers when reidrected to or from
        the NASA authorisation host.
        """

        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (
                original_parsed.hostname != redirect_parsed.hostname
                and redirect_parsed.hostname != self.AUTH_HOST
                and original_parsed.hostname != self.AUTH_HOST
            ):
                del headers['Authorization']


def create_earthdata_session() -> requests.Session:
    """
    
    """

    # Load in pipeline configuration
    pipeline_config = load_config()

    # Return instantiated earth data session
    return EarthdataSession(
        pipeline_config.login_credentials.username, 
        pipeline_config.login_credentials.password
        )

