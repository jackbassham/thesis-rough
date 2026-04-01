from datetime import date, timedelta
from typing import Generator

class DatasetURLBuilder:
    """
    
    """

    def __init__(
        self,
        parent_url: str,
        dataset_directory_template: str,
        filename_template: str,
        hemisphere_directory_map: None | dict[str, str],
        hemisphere_filename_map: dict[str, str],
    ):

        self.parent_url = parent_url
        self.dataset_directory_template = dataset_directory_template,
        self.filename_template = filename_template,
        self.hemisphere_directory_map = hemisphere_directory_map,
        self.hemisiphere_filename_map = hemisphere_filename_map,

    def construct_all_urls(
            self, 
            hemisphere: str,
            start_year: int,
            end_year: int,
            freq: str
    ) -> Generator[str]:




