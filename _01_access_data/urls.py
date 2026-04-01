from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Generator


# Define template abstract base class
class URLBuilder(ABC):

    def __init__(self, config):
        self.config = config

    # Extract data configuration parameters for URLs as properties
    @property
    def data_config(self):
        return self.config.data_config
    
    @property
    def hemisphere(self) -> str:
        return self.data_config.hemisphere
    
    @property
    def start_year(self) -> int:
        return self.data_config.year_range[0]
    
    @property
    def end_year(self) -> int:
        return self.data_config.year_range[1]

    @abstractmethod
    def build(self) -> Generator[str, None, None]:
        pass


# Define child classes for each data set

class IceVelURLBuilder(URLBuilder):
    """
    Polar Pathfinder Daily 25 km EASE-Grid Sea Ice Motion Vectors, Version 4
    DATA SET ID: NSIDC-0116
    DOI: 10.5067/INAWUWO7QH7B
    https://nsidc.org/data/nsidc-0116/versions/4

    Example url:
    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/south/daily/
    Example filename:
    icemotion_daily_sh_25km_19900101_19901231_v4.1.nc

    NOTE Subject to change, Nasa Earth Data moving to cloud.
    """

    # Define base url for datset download directories
    BASE_URL = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/'
    
    # Define mapping dict from hemisphere parameter to dataset labels
    HEM_MAP_DIR = {'south': 'south', 'north': 'north'}
    HEM_MAP_FILE = {'south': 'sh', 'north': 'nh'}

    def build(self):

        # Get hemisphere from mapping
        hem_dir = self.HEM_MAP_DIR[self.hemisphere]
        hem_file = self.HEM_MAP_FILE[self.hemisphere]

        # Define parent directory
        parent = f'{self.BASE_URL}/{hem_dir}/daily'

        for year in range(self.start_year, self.end_year + 1):
            # Generate filenames for current year's file
            filename = f'icemotion_daily_{hem_file}_25km_{year}0101_{year}1231_v4.1.nc'

            # Yield full URL for current year's data download
            yield(f'{parent}/{filename}')


class IceConcURLBuilder:
    """
    Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS Passive Microwave Data, Version 2
    DATA SET ID: NSIDC-0051
    DOI: 10.5067/MPYG15WAA4WX
    https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3177837840-NSIDC_CPRD
    NOTE It seems that Nimbus-7 sea ice concentrations only go until 2024.
    2023 - Present Concentrations are in this dataset;
    AMSR2 Daily Polar Gridded Sea Ice Concentrations, Version 2
    https://nsidc.org/data/nsidc-0803/versions/2


    NOTE 
    New url:
    https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3177837840-NSIDC_CPRD/temporal/1989/01/01
    New filename:
    NSIDC0051_SEAICE_PS_S25km_19890101_v2.0.nc
    """

    # Define base url for datset download directories
    BASE_URL = 'https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3177837840-NSIDC_CPRD/temporal/'

    # Define mapping dict from hemisphere parameter to dataset labels
    HEM_MAP_FILE = {'south': 'S', 'north': 'N'}

    def build(self):

        # Start URLs at first day of start year
        start_date = date(self.start_year, 1, 1)

        # End URLs at last day of end year
        end_date = date(self.end_year, 12, 31)

        # Get hemisphere from mapping
        hem_file = self.HEM_MAP_FILE[self.hemisphere]

        # Initialize current date
        current_date = start_date
        
        while current_date <= end_date:
            # Get current year, month, day
            yyyy = current_date.strftime('%Y')
            mm = current_date.strftime('%m')
            dd = current_date.strftime('%d')

            # Yield full URL for current day's data download
            yield(
                # Base and directory
                f'{self.BASE_URL}/temporal/{yyyy}/{mm}/{dd}/'
                # Filename
                f'NSIDC0051_SEAICE_PS_{hem_file}25km_{yyyy}{mm}{dd}_v2.0.nc'
            )

            # Move to next day
            current_date += timedelta(days=1)


# TODO
class WindURLBuilder:
    
    # TODO implement url for wind data
    # NOTE try JRA55, ERA, or ECMFW?
    ...