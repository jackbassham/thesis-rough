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


def build_complex_features(x):
    """
    
    """

    # Unpack features
    ua_t0, va_t0, ci_t1 = x[:,0], x[:,1], x[:,2]
    
    # TODO include mask values as weights in LR
    # NOTE Going to experiment with including mask in both LR and CNN
    # mask = x[:,3]

    # Make complex features
    za_t0 = ua_t0 + va_t0*1j 
    zci_t1 = ci_t1 + ci_t1*1j

    # Create list of features
    features = [
        za_t0, # Complex pesent day day wind vector
        zci_t1, # Complex previous day ice concentration
    ]

    # Stack features into feature matrix columns (number_samples, number_features)
    X = np.stack(features, axis=1)

    # Add a column of ones along feature axis to represent the constant parameter
    X = np.concatenate(
        [X, np.ones((X.shape[0], 1), dtype=complex)],
        axis=1
    )

    return X


def build_complex_targets(y):
    """
    
    """
    return y[:,0] + y[:,1]*1j # Complex previous day ice velocity vector


def build_complex_uncertainty_weights(r, epsilon=1e-4):
    """
    
    """

    # Convert squared uncertainty to complex
    # NOTE squared uncertainty used for weighting
    # z_r**2 = (r_u + ir_v)(r_u - ir_v)
    #      = r_u**2 + r_v**2
    #   if r_u = r_v = r, z_r**2 = 2r**2  
    zr_squared = 2 * r ** 2

    # Compute model weights as inverse uncertainty squared + small correction
    w = 1 / (zr_squared + epsilon)

    return w