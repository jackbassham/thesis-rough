import numpy as np
import numpy.typing as npt


def chronological_indices(
        time: npt.NDArray[np.datetime64],
        n_val_years: int = 2, n_test_years: int = 2
        ) -> list[dict[str, npt.NDArray[np.floating]]]:
    """
    
    """

    # Get array of n_time years from the time array
    years = time.astype('datetime64[Y]')

    # Get array of unique years for the splits
    unique_years = np.unique(years)

    # Check that years will work for split
    validate_split_years(unique_years, n_val_years, n_test_years)

    # The last 'n_test' years make test split
    test_years = unique_years[-n_test_years:]
    # The next 'n_val' years make the validation split
    val_years = unique_years[-(n_test_years+ n_val_years):-n_test_years]
    # The remaining years in range make the training split
    train_years = unique_years[:-(n_test_years+ n_val_years)]

    # Create dict of split indices arrays, adding extra dimension to represent the member axis
    split_indices = {
            'test': np.expand_dims(np.where(np.isin(years, test_years))[0], axis=0),
            'val': np.expand_dims(np.where(np.isin(years, val_years))[0], axis=0),
            'train': np.expand_dims(np.where(np.isin(years, train_years))[0], axis=0)  
        }     

    return split_indices


def k_randomized_year_indices(
        time: npt.NDArray[np.datetime64],
        n_members: int = 10,
        n_val_years: int = 2, n_test_years: int = 2,
        seed: int = 0
        ) -> list[dict[str, npt.NDArray[np.floating]]]:
    """
    
    """

    # Get array of n_time years from the time array
    years = time.astype('datetime64[Y]')

    # Get array of unique years for the splits
    unique_years = np.unique(years)

    # Check that years will work for split
    validate_split_years(unique_years, n_val_years, n_test_years)

    # Initialize empty lists for ensemble member split indices
    test_indices = []
    val_indices = []
    train_indices = []

    # Initialize random number generator with seed
    rng = np.random.default_rng(seed)

    for _ in range(n_members):
        
        # Randomly permutate the array of unique years
        shuffled_years = rng.permutation(unique_years)

        # The last 'n_test' years make test split
        test_years = shuffled_years[-n_test_years:]
        # The next 'n_val' years make the validation split
        val_years = shuffled_years[-(n_test_years + n_val_years):-n_test_years]
        # The remaining years in range make the training split
        train_years = shuffled_years[:-(n_test_years + n_val_years)]

        # Append member's indices to the each split list
        test_indices.append(np.where(np.isin(years, test_years))[0])
        val_indices.append(np.where(np.isin(years, val_years))[0])
        train_indices.append(np.where(np.isin(years, train_years))[0])  

    # Return dict of split arrays with each member joined along the first axis
    split_indices = {
        'test': np.stack(test_indices, axis=0),
        'val': np.stack(val_indices, axis=0),
        'train': np.stack(train_indices, axis=0)  
    }

    return split_indices


# TODO k_fold_sliding_indices


def validate_split_years(
    years: npt.NDArray[np.datetime64], 
    n_val_years: int, n_test_years: int) -> None:
    """
    
    """

    # Handle case where not enough years to split
    if len(np.unique(years)) <= n_val_years + n_test_years:
        raise ValueError(
            'Not enough years in data for split: '
            f'{len(np.unique(years))} years in data, {n_val_years} in val split, {n_test_years} in test split'
        )



