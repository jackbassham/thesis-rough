from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import warnings


@dataclass
class SplitIndices:

    # Construct parameters
    test: npt.NDArray
    val: npt.NDArray
    train: npt.NDArray


def chronological_split(
        time: npt.NDArray[np.datetime64],
        n_val: int = 2, n_test: int = 2) -> SplitIndices:
    """
    
    """

    # Get array of n_time years from the time array
    years = time.astype('datetime64[Y]')

    # Check that years will work for split
    validate_split_years(years)

    # Create array of unique years in split
    unique_years = np.unique(years)

    # The last 'n_test' years make test split
    test_years = unique_years[-n_test:]
    # The next 'n_val' years make the validation split
    val_years = unique_years[-(n_test + n_val):-n_test]
    # The remaining years in range make the training split
    train_years = unique_years[:-(n_test + n_val)]

    # Get split indices where data years in split years
    split_indices = SplitIndices(
        test = np.where(np.isin(years, test_years))[0],
        val = np.where(np.isin(years, val_years))[0],
        train = np.where(np.isin(years, train_years))[0]
    )

    return split_indices


def randomized_split():
    """
    
    """

    # TODO generate seed in config

    ...


def ensemble_spllit():
    """
    
    """

    ...


def validate_split_years(
        years: npt.NDArray[np.datetime64], 
        n_val: int, n_test: int) -> None:
    """
    
    """

    # Handle case where not enough years to split
    if len(np.unique(years)) <= n_val + n_test:
        raise ValueError(
            'Not enough years in data for split: '
            f'{len(np.unique(years))} years in data, {n_val} in val split, {n_test} in test split'
        )



