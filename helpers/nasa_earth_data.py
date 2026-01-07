import requests
import io

from .param import USER, PASS

def get_temp_NED_file(url):
    """Gets temporary file from Nasa Earth Data Website via URL"""
    ### Create session for NASA Earthdata ###
    # Overriding requests.Session.rebuild_auth to mantain authentication headers when redirected
    # Custom subclass to extend functionality of parent class requests.session to maintain authentication headers
    # when server redirects requests
    class SessionWithHeaderRedirection(requests.Session):
        # Host for which authentication headers maintained
        AUTH_HOST = 'urs.earthdata.nasa.gov'
    
        # Define 'costructor method' for sublclass SessionWithHeaderRedirection  
        # Called to initialze attributes of subclass when created (here with parameters username and password)
        # 'self' parameter is in reference to the instance constructed (our subclass)
        def __init__(self, username, password):
            # Call constructor method for parent class (executing initialization code defined in parent class)
            super().__init__()
            # Initialize authentication atribute in class containing username and password
            self.auth = (username, password)

        # Overrides from the library to keep headers when redirected to or from the NASA auth host
        def rebuild_auth(self, prepared_request, response):
            headers = prepared_request.headers
            url = prepared_request.url

            if 'Authorization' in headers:
                original_parsed = requests.utils.urlparse(response.request.url)
                redirect_parsed = requests.utils.urlparse(url)

                if (original_parsed.hostname != redirect_parsed.hostname) and \
                        redirect_parsed.hostname != self.AUTH_HOST and \
                        original_parsed.hostname != self.AUTH_HOST:
                    del headers['Authorization']

    # Create session with the user credentials that will be used to authenticate access to the data
    session = SessionWithHeaderRedirection(USER, PASS)

    try:
        # submit the request using the session
        response = session.get(url, stream=True)
        # '200' means success
        StatusCode = response.status_code
        print(StatusCode)
        # raise an exception in case of http errors
        response.raise_for_status()  

        # Read response content to temp using BytesIO object
        temp = io.BytesIO(response.content)

        return temp

    except requests.exceptions.HTTPError as e:
        # Handle any errors here
        print(f"HTTP Error: {e}")
        return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None