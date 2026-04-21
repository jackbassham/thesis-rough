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


def make_complex_inputs(data: dict[str, npt.NDArray[np.float32]], include_uncertainty: bool=False):
    """
    
    """

    # For all arrays in data
    for name, array in data.items():
        # Convert nan values to 0
        data[name] = np.nan_to_num(array, nan=0.0)

    # Unpack features
    ua_t0 = data['x'][:,0,:,:]
    va_t0 = data['x'][:,1,:,:]
    ci_t1 = data['x'][:,2,:,:]
    # TODO include mask values as weights in LR
    # NOTE Going to experiment with including mask in both LR and CNN
    mask = data['x'][:,3,:,:]

    # Unpack targets
    ui_t0 = data['y'][:,0,:,:]
    vi_t0 = data['y'][:,1,:,:]

    # Make complex features
    za_t0 = ua_t0 + va_t0*1j # Complex pesent day day wind vector
    zci_t1 = ci_t1 + ci_t1*1j # Complex previous day ice concentration

    # Make complex targets
    zi_t0 = ui_t0 + vi_t0*1j # Complex previous day ice velocity vector

    if include_uncertainty:
        # Unpack uncertainty
        ri_t0 = np.squeeze(data['ri_t0'], axis=1)

        # Convert squared uncertainty to complex
        # NOTE squared uncertainty used for weighting
        # z_r**2 = (r_u + ir_v)(r_u - ir_v)
        #      = r_u**2 + r_v**2
        #   if r_u = r_v = r, z_r**2 = 2r**2  
        zri_t0 = 2 * ri_t0 ** 2

        return za_t0, zci_t1, zi_t0, zri_t0
    
    else: 
        return za_t0, zci_t1, zi_t0


def build_gram_and_data_matrices(targets: tuple, features: tuple, uncertainty: np.ndarray | None = None):



    # Stack features into gram matrix along channel dimensions
    # NOTE last column is constant and consists of ones, same dtype as features
    G = np.stack(
        [features, np.ones_like(features[0])],
        axis=1
    )
    
    # Transpose targets into data matrix
    d = targets.T

    if uncertainty is not None:
        # NOTE uncertainty argument for weighting is complex, squared uncertainty
        W = 1 / (uncertainty + 1e-4)

        # NOTE, applying weights early on to avoid diags() memory useage
        G_w = G * W[:, None]
        d_w = d * W

        return G_w, d_w
    
    else:
        return G, d