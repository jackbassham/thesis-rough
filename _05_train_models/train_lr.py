import numpy as np
from pathlib import Path
from . import utils_lr
from . import models

# Define model type string for saving predictions
MODEL_STR = 'lr_cf'

def main(cfg):

    # Load model inputs source path
    path_model_inputs = cfg.path_config.data_stage_path('model_inputs')

    # Load in training data for fit
    train = np.load(path_model_inputs / 'train.npz')
    # Get features and targets from data, excluding mask (last feature)
    x_train, y_train = train['x'][:,:-1,:,:], train['y']

    # Instantiate model with functions to build complex features and targets
    model = models.UnweightedLR(
        feature_fcn = utils_lr.build_complex_features,
        target_fcn = utils_lr.build_complex_targets,
    )

    # Perform fit and solve for coefficients
    model.fit(x_train, y_train)

    # Get array of real coefficients from model
    R_coef = model.R_coef_() # (n_Re * n_Im, height, width)

    # Load model output destination path
    path_out = cfg.path_config.model_path('lr_cf')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_out)

    # Save coefficinets
    # TODO coefficient labels/ names in dict?
    np.savez(
        path_out / 'coeffients.npz',
        R_coef
    )

    # Load in testing data for predictions
    test = np.load(path_model_inputs / 'test.npz')
    # Get features and targets from data, excluding mask (last feature)
    x_test, y_test = test['x'][:,:-1,:,:], test['y']

    # Get complex predictions on test split from the fit model
    Z_preds = model.predict(x_test)

    # Convert complex predictions to real for saving and evaluation
    R_preds = model.z_to_vector(Z_preds)

    # Save predictions and true values
    # NOTE using y_test for true now, move to just saving predictions
    np.savez(
        path_out / 'preds.npz',
        y_pred = R_preds,
        y_true = y_test
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)

