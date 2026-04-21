import numpy as np


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


def make_lr_inputs():
    """
    
    """
    