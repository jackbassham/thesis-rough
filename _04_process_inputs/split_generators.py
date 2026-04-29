import numpy as np
import numpy.typing as npt
from pathlib import Path


def chronological_indices(
        time: npt.NDArray[np.datetime64],
        n_members: int = 1,
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


def k_shuffled_year_indices(
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

    for unique_year in unique_years:
        print('uique_year: ', unique_year)
        print(np.shape(np.where(np.isin(years, unique_year))[0]))

    # Check that years will work for split
    validate_split_years(unique_years, n_val_years, n_test_years)

    # Initialize empty lists for ensemble member split indices
    test_indices, val_indices, train_indices = [], [], []

    # Initialize empty lists for ensemble member split years metadata
    test_years_meta, val_years_meta, train_years_meta = [], [], []

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

        # Append test, train, and val years used to metadata lists
        test_years_meta.append(test_years)
        val_years_meta.append(val_years)
        train_years_meta.append(train_years)

        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'test_years: {test_years}')
        print(f'val_years: {val_years}')
        print(f'train_years: {train_years}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        print('')

        # Append member's indices to the each split list
        test_indices.append(np.where(np.isin(years, test_years))[0])
        val_indices.append(np.where(np.isin(years, val_years))[0])
        train_indices.append(np.where(np.isin(years, train_years))[0])  

        for idx_array in test_indices:
            print(f'test shape idx: {np.shape(idx_array)}')

    # Create dict of splits with each split containing list of member indices arrays
    split_indices = {
        'test': test_indices,
        'val': val_indices,
        'train': train_indices  
    }

    # Create dict of metadata for each split
    split_years_meta = {
        'test': test_years_meta,
        'val': val_years_meta,
        'train': train_years_meta,
    }

    return split_indices, split_years_meta


def save_member_split_indices(
        path: Path,
        split_indices: dict[str, list[npt.NDArray]],
        split_years_meta: dict[str, list[npt.NDArray]] | None = None,
    ) -> None:

    """
    
    """

    # Loop through each split's list of ensemble member indices arrays
    for split_name, member_arrays in split_indices.items():

        # Create a dict of indices arrays with keys reflecting each ensemble member
        indices = {
            f'{m:02d}': array for m, array in enumerate(member_arrays)
        }

        # Save the dict of member arrays for that split
        np.savez(path / f'indices_{split_name}.npz', **indices)

    if split_years_meta is not None:
        # Loop through each split's list of member split years arrays
        for split_name, split_years in split_indices.items():

            # Create dict of split years as meta data
            meta = {
                f'{m:02d}': array for m, array in enumerate(split_years)
            }

            # Save the dict of split years meta data
            np.savez(path / f'split_years_meta_{split_name}', **meta)


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



