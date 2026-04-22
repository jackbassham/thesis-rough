from . import models
import numpy as np
from . import lr_utils

# Define model type string for saving predictions
MODEL_STR = 'lr_cf'

def main(cfg):

    # Load model inputs source path
    path_model_inputs = cfg.path_config.data_stage_path('model_inputs')

    # Load in training data for fit
    train = np.load(path_model_inputs / 'train.npz')
    # Get features and targets from data, excluding mask (last feature)
    x_train, y_train = train['x'][:,:-1,:,:], train['y'][:,:-1,:,:]  


    # Load in testing data for fit
    test = np.load(path_model_inputs / 'test.npz')
    # Get features and targets from data, excluding mask (last feature)
    x_test, y_test = test['x'][:,:-1,:,:], test['y'][:,:-1,:,:]    

    # Instantiate model
    model = models.UnweightedLR(
        lr_utils.build_complex_features,
        lr_utils.build_complex_targets,
    )

    # Perform fit and solve for coefficients
    model.fit(x_train, y_train)




    # Train model
    zm, zfit_tr, ztrue_tr = lr_train(x_train, y_train)

    # Initialize arrays for real training outputs
    # NOTE two extra coefficients for u and v mean
    m = np.full((nm, nlat, nlon), np.nan) # model coefficients, real
    fit_tr = np.full((nt_tr, nout, nlat, nlon), np.nan) # training fit, real
    true_tr = np.full((nt_tr, nout, nlat, nlon), np.nan) # training true, real

    # TODO make loop or use advanced indexing for real and imaginary coefficients

    # Convert training coefficients to real
    m[0, :, :] = zm[0, :, :].real # C_uproj, (constant)
    m[1, :, :] = zm[0, :, :].imag # C_vproj, (constant)
    m[2, :, :] = zm[1, :, :].real # A_uproj, (ua_t0)
    m[3, :, :] = zm[1, :, :].imag # A_uproj, (va_t0)
    m[4, :, :] = zm[2, :, :].real # B_uproj, (ci_t1)
    m[5, :, :] = zm[2, :, :].imag # B_uproj, (ci_t1)

    # Convert training fit to real
    fit_tr[:, 0, :, :] = zfit_tr.real # ui_t0, fit
    fit_tr[:, 1, :, :] = zfit_tr.imag # vi_t0, fit

    # Convert training true to real
    true_tr[:, 0, :, :] = ztrue_tr.real # ui_t0, true
    true_tr[:, 1, :, :] = ztrue_tr.imag # vi_t0, true

    # Create the destination directory if it doesn't already exist
    os.makedirs(PATH_LR_CF_OUT, exist_ok = True)

    # Save coeffients, fit
    np.savez(
        os.path.join(PATH_LR_CF_OUT, f"coef_fit_{MODEL_STR}.npz"),
        m = m,
        fit_tr = fit_tr,
        true_tr = true_tr,
    )

    # Get predictions on test set
    zpred_te, ztrue_te = lr_test(x_test, y_test, zm)

    # Intialize arrays for test output predictions
    # NOTE y notation used for consistency with CNN and plotting
    y_pred = np.full((nt_te, nout, nlat, nlon), np.nan) 
    y_true = np.full((nt_te, nout, nlat, nlon), np.nan)

    # Convert test predictions to real
    y_pred[:,0,:,:] = zpred_te.real # ui_t0, pred
    y_pred[:,1,:,:] = zpred_te.imag # vi_t0, pred

    # Convert test true to real
    y_true[:,0,:,:] = ztrue_te.real # ui_t0, true
    y_true[:,1,:,:] = ztrue_te.imag # vi_t0, true

    # Save predictions
    np.savez(
        os.path.join(PATH_LR_CF_OUT, f"preds_{MODEL_STR}.npz"),
        y_pred = y_pred,
        y_true = y_true,
    )

    return



if __name__ == "__main__":
    main()

