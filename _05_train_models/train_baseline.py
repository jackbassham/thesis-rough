import numpy as np
from . import(
    ensemble,
    models
)

def main(cfg):

    # Load in current member's split targets, features
    splits = ensemble.load_member_splits(cfg)

    # Load test split for current member
    y_test = splits['test']['y']

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
    from .ensemble import run_ensemble
    cfg = load_config()
    run_ensemble(cfg, main)

