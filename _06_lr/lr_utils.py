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


def build_complex_gram_data_matrices(data: dict[str, npt.NDArray[np.float32]], weighted_by_uncertainty: bool=False):
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

    # Stack features into gram matrix along channel dimensions
    # NOTE last column is constant and consists of ones of complex input dtype
    G = np.stack(
        [za_t0, zci_t1, np.ones_like(za_t0)]
    )
    
    # Transpose targets into data matrix
    d = zi_t0.T

    if weighted_by_uncertainty:
        # Unpack uncertainty
        ri_t0 = np.squeeze(data['ri_t0'], axis=1)

        # 

