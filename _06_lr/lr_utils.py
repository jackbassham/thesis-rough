import numpy as np
import numpy.typing as npt


def load_input_data_from_npz(config):
    """
    
    """

    # Load model inputs source path
    path_model_inputs = config.path_config.data_stage_path('model_inputs')

    # Return train and test npz data
    train = np.load(path_model_inputs / 'train.npz')
    test = np.load(path_model_inputs / 'test.npz')

    # Get inputs from npz data
    x_train, y_train, mask_train = train['x'], train['y'], train['mask']
    x_test, y_test, mask_test = test['x'], test['y'], test['mask']


def make_complex_features_targets(x, y):
    """
    
    """

    # Convert nans to zeros
    # NOTE filling nan with zero now to reflect Hoffman and CNN
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Unpack features
    ua_t0, va_t0, ci_t1 = x[:,0], x[:,1], x[:,2]
    # TODO include mask values as weights in LR
    # NOTE Going to experiment with including mask in both LR and CNN
    mask = x[:,3]

    # Make complex features
    za_t0 = ua_t0 + va_t0*1j # Complex pesent day day wind vector
    zci_t1 = ci_t1 + ci_t1*1j # Complex previous day ice concentration

    # Unpack targets
    ui_t0, vi_t0 = y[:,0], y[:,1]

    # Make complex targets
    zi_t0 = ui_t0 + vi_t0*1j # Complex previous day ice velocity vector

    return za_t0, zci_t1, zi_t0


def get_input_dimensions(input: dict[str, npt.NDArray]):
    """
    
    """

    # Get dimensions from first input assuming shape (time, channel, height, width)
    in_channels, height, width = np.shape(next(iter(input.values())))[1:]

    return(
        in_channels,
        height,
        width
    )


def make_complex_weights(uncertainty):
    """
    
    """

    # Convert squared uncertainty to complex
    # NOTE squared uncertainty used for weighting
    # z_r**2 = (r_u + ir_v)(r_u - ir_v)
    #      = r_u**2 + r_v**2
    #   if r_u = r_v = r, z_r**2 = 2r**2  
    uncertainty = 2 * uncertainty ** 2

    return uncertainty


def build_gram_and_data_matrices(targets: tuple, features: tuple, uncertainty: np.ndarray | None = None):



    # Stack features into gram matrix along channel dimensions
    # NOTE last column is constant and consists of ones, same dtype as features
    G = np.stack(
        [features, np.ones_like(features[0])],
        axis=1
    )
    
    # Transpose targets into data matrix
    d = targets.T
    
    return (G, d)