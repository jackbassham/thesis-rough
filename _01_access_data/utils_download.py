from dataclasses import dataclass


@dataclass
class NsidcDataset:

    # Define parent url
    parent: str

    dataset: str

    filename: str