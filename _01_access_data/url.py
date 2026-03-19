from dataclasses import dataclass
from datetime import datetime, timedelta

# TODO Possible dataclass structure for entire datset - download 
# trough preprocessing?


@dataclass
class IceVelURL:
    """
    Polar Pathfinder Daily 25 km EASE-Grid Sea Ice Motion Vectors, Version 4
    DATA SET ID: NSIDC-0116
    DOI: 10.5067/INAWUWO7QH7B
    https://nsidc.org/data/nsidc-0116/versions/4
    """

    parent_url: str
    file_pattern: str
    hemisphere_map: dict




@dataclass
class IceConcURL:
    """
    Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS Passive Microwave Data, Version 2
    DATA SET ID: NSIDC-0051
    DOI: 10.5067/MPYG15WAA4WX
    https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3177837840-NSIDC_CPRD
    NOTE It seems that Nimbus-7 sea ice concentrations only go until 2024.
    2023 - Present Concentrations are in this dataset;
    AMSR2 Daily Polar Gridded Sea Ice Concentrations, Version 2
    https://nsidc.org/data/nsidc-0803/versions/2

    Parent url:
    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/south/daily/
    filename:
    icemotion_daily_sh_25km_19900101_19901231_v4.1.nc


    NOTE 
    New root url:
    https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3177837840-NSIDC_CPRD/temporal/1989/01/01
    New filename:
    NSIDC0051_SEAICE_PS_S25km_19890101_v2.0.nc
    """



    parent_url: str
    file_pattern: str
    hemisphere_map: dict



@dataclass
class WindURL:
    
    # TODO implement url for wind data
    # NOTE try JRA55, ERA, or ECMFW?
    ...