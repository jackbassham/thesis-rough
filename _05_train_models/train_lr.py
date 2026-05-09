import numpy as np
from pathlib import Path
from . import (
    ensemble,
    models,
    utils_lr
) 

# Define model type string for saving predictions
MODEL_STR = 'lr_cf'

def main(cfg):

    # Load in current member's split targets, features
    splits = ensemble.load_member_splits(cfg)

    # Get features and targets from training splits, excluding mask (last feature)
    x_train = splits['train']['x']
    y_train = splits['train']['y']

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

    # Get features and targets from training splits, excluding mask (last feature)
    x_test = splits['test']['x'][:,:-1,:,:]
    y_test = splits['test']['y']

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
    from .ensemble import run_ensemble
    cfg = load_config()
    run_ensemble(cfg, main)

