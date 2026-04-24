import numpy as np
import os
from . import models

def main(cfg):

    # Load model inputs source path
    path_model_inputs = cfg.path_config.data_stage_path('model_inputs')

    # Load in testing data for predictions
    test = np.load(path_model_inputs / 'test.npz')
    # Get targets from data, excluding mask (last feature)
    y_test = test['y']

    # Instantiate one day persistence baseline model
    model = models.PersistenceBaseline()

    # Get predictions on test splilt
    y_pred = model.predict(y_test)

    # Load model output destination path
    path_out = cfg.path_config.model_path('ps')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_out)

    # Save predictions
    np.savez(
        path_out / 'preds.npz',
        y_pred = y_pred,
        y_true = y_test,
    )


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)

