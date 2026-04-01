from datetime import date, timedelta
from typing import Generator

class DatasetURLBuilder:
    """
    
    """

    def __init__(
        self,
        parent: str,
        dataset_directory_template: str,
        filename_template: str,
        hemisphere_directory_map: None | dict[str, str],
        hemisphere_filename_map: dict[str, str],
    ):

        self.parent = parent
        self.dataset_directory_template = dataset_directory_template,
        self.filename_template = filename_template,
        self.hemisphere_directory_map = hemisphere_directory_map,
        self.hemisiphere_filename_map = hemisphere_filename_map,

    def build_urls_list(
            self, 
            hemisphere: str,
            start_date: date,
            end_date: date,
            file_freq: str
    ) -> Generator[str]:
        
        # Initialize empty url list
        urls = []

        # Handle case where data is stored in daily files
        if file_freq.lower.strip() == 'daily':
            # Compute data temporal range
            time_range = end_date - start_date

            # Iterate through days in temporal range
            for d in range(time_range.days + 1):
                # Get day for current iteration
                day = start_date + timedelta(days = d)
                # Build url for current day and append to list
                urls.append(self._build_url(hemisphere, day))

        # Handle case where data is stored in monthly files
        elif file_freq.lower.strip() == 'monthly':


        
                


    def _build_url(
            self,
            hemisphere: str,
            date_time: date,
    ) -> str:





