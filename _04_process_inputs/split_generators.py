from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


def chronological_indices(
        time: npt.NDArray[np.datetime64],
        n_val: int = 2, n_test: int = 2
        ) -> list[dict[str, npt.NDArray[np.floating]]]:
    """
    
    """

    # Get array of n_time years from the time array
    years = time.astype('datetime64[Y]')

    # Get array of unique years for the splits
    unique_years = np.unique(years)

    # Check that years will work for split
    validate_split_years(unique_years, n_val, n_test)

    # Initialize empty list for split indices for downstream code (there will only be one member here)
    split_indices = []

    # The last 'n_test' years make test split
    test_years = unique_years[-n_test:]
    # The next 'n_val' years make the validation split
    val_years = unique_years[-(n_test + n_val):-n_test]
    # The remaining years in range make the training split
    train_years = unique_years[:-(n_test + n_val)]

    # Fill list with split indices dict
    split_indices = [
        {
            'test': np.where(np.isin(years, test_years))[0],
            'val': np.where(np.isin(years, val_years))[0],
            'train': np.where(np.isin(years, train_years))[0]  
        }     
    ]

    return split_indices


def k_randomized_year_indices(
        time: npt.NDArray[np.datetime64],
        n_members: int = 10,
        n_val: int = 2, n_test: int = 2,
        seed: int = 0
        ) -> list[dict[str, npt.NDArray[np.floating]]]:
    """
    
    """

    # Get array of n_time years from the time array
    years = time.astype('datetime64[Y]')

    # Get array of unique years for the splits
    unique_years = np.unique(years)

    # Check that years will work for split
    validate_split_years(unique_years, n_val, n_test)

    # Initialize empty list for ensemble member split indices
    split_indices = []

    # Initialize random number generator with seed
    rng = np.random.default_rng(seed)

    for member in range(n_members):

        # TODO label members
        
        # Randomly permutate the array of unique years
        shuffled_years = rng.permutation(unique_years)

        # The last 'n_test' years make test split
        test_years = shuffled_years[-n_test:]
        # The next 'n_val' years make the validation split
        val_years = shuffled_years[-(n_test + n_val):-n_test]
        # The remaining years in range make the training split
        train_years = shuffled_years[:-(n_test + n_val)]

        # Append member's split indices to the list
        split_indices.append({
            'test': np.where(np.isin(years, test_years))[0],
            'val': np.where(np.isin(years, val_years))[0],
            'train': np.where(np.isin(years, train_years))[0]  
        })   

    return split_indices

    




# TODO k_fold_sliding_indices


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



