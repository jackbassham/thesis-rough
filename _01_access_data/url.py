from dataclasses import dataclass
from datetime import datetime, timedelta


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
    """

    ...



@dataclass
class WindURL:
    
    # TODO implement url for wind data
    ...