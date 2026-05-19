from numpy import np
from pathlib import Path


def rescale_predictions(config, y):

    # Load path to statistics for rescaling from configuration
    path_stats = config.path_config.data_stage_path('mask_norm')

    # Load standard deviation of ice speed for rescaling
    Ui_t0 = np.load(path_stats / 'global_stds.npz')['Ui_t0']

    # Load gridwise means for rescaling
    gridwise_means = np.load(path_stats / 'gridwise_means.npz')

    # Check that y is shaped (time, channel, height, width)
    if np.ndims(y) != 4:
        raise ValueError(
            f'Invalid number of input dimensions for rescaling: {np.ndims(y)} dims'
            f'Rescaling input "y" needs shape (time, channel, height, width)'
        )
    
    y_rescaled = np.full_like(y, np.nan)
    
    # Rescale u predictions (first channel)
    y_rescaled[:,0] = (y[:,0] * Ui_t0) + gridwise_means['ui_t0']
    # Rescale v predictions (second channel)
    y_rescaled[:,1] = (y[:,1] * Ui_t0) + gridwise_means['vi_t0']

    return y_rescaled