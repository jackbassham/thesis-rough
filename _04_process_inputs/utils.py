import numpy.typing as npt


def make_target_feature_arrays(inputs: dict[str, npt.NDArray]
                               ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    
    """

    # Define target variables
    targets = [
        'ui_t0',
        'vi_t0',
    ]

    # Define feature variables
    features = [
        'ua_t0',
        'va_t0',
        'ci_t1',
        'mask'
    ]

    # Get input dimensions from first input
    nt, nlat, nlon = next(iter(inputs.values())).shape

    # Initialize target, feature arrays
    y = np.zeros((nt, len(targets), nlat, nlon))
    x = np.zeros((nt, len(features), nlat, nlon))

    # Fill target array
    for i, name in enumerate(targets):
        y[:, i] = inputs[name]

    # Fill feature array
    for i, name in enumerate(features):
        x[:, i] = inputs[name]

    # Convert nan values in targets, features to zero and float32()
    # NOTE PyTorch default for model is FLoat (np.float32 equivalent) 
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)

    # Add channel dimension on uncertainty to match targets, features
    ri_t0 = inputs['ri_t0'][:, np.newaxis, :, :]

    # Convert nan values in uncertainty to 1000 (flag) and float32()
    ri_t0 = np.nan_to_num(ri_t0, nan=1000.0).astype(np.float32)

    print(y.shape)
    print(x.shape)

    return {
        'y': y,
        'x': x,
        'ri_t0': ri_t0
    }


def split_arrays(
        arrays: dict[str, npt.NDArray[np.floating]], 
        indices: dict[str, npt.NDArray[np.floating]],
    ) -> dict[str, dict[str, npt.NDArray[np.floating]]]:
    """
    
    """

    # Initialize train, val, and test split dics
    splits = {
        'train': {},
        'val': {},
        'test': {},
    }

    for name, array in arrays.items():
        splits['train'][name] = array[indices['train']]
        splits['val'][name] = array[indices['val']]
        splits['test'][name] = array[indices['test']]

    return splits