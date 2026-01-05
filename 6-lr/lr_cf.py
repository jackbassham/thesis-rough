import numpy as np
from numpy import linalg as LA
import os

from .path import(
    PATH_SOURCE,
    PATH_DEST,
    FSTR_END_IN,
    FSTR_END_OUT,
)

# Define model type string for saving predictions
MODEL_STR = 'lr_cf'

def main():
    
    # Load in training data
    data = np.load(os.path.join(PATH_SOURCE,f'train_{FSTR_END_IN}.npz'))
    x_train = data['x_train']
    y_train = data['y_train']

    # Load in testing data
    data = np.load(os.path.join(PATH_SOURCE,f'test_{FSTR_END_IN}.npz'))
    x_test = data['x_test']
    y_test = data['y_test']

    # Get train batch dimensions
    nt_tr, nout, nlat, nlon = np.shape(y_train)

    # Get input channel dimensions
    _, nin, _, _, = np.shape(x_train)

    # TODO make coefficient dimension dynamic

    # Get coefficient dimension
    nm = nin + 1 # NOTE +1 for u and v complex concentration projections 
    nm = nm + 2 # NOTE +2 for u and v constants (mean ~ 0) (constant column G[0])

    # Get test batch dimensions
    nt_te, _, _, _ = np.shape(y_test)

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
    true_tr[:, 1, :, :] = ztrue.tr.imag # vi_t0, true

    # Save coeffients, fit
    np.savez(
        os.path.join(PATH_DEST, f"coef_fit_{MODEL_STR}_{FSTR_END_OUT}.npz"),
        m = m,
        fit_tr = fit_tr,
        true_tr = true_tr,
    )

    # Get predictions on test set
    zpred_te, ztrue_te = lr_test(x_test, y_test, zm)

    # Intialize arrays for test output predictions
    # NOTE y notation used for consistency with CNN and plotting
    y_pred = np.full((nt_te, nout, nlat, nlon), np.nan) 
    y_true = y_pred

    # Convert test predictions to real
    y_pred[:,0,:,:] = zpred_te.real # ui_t0, pred
    y_pred[:,1,:,:] = zpred_te.imag # vi_t0, pred

    # Convert test true to real
    y_true[:,0,:,:] = ztrue_te.real # ui_t0, true
    y_true[:,1,:,:] = ztrue_te.imag # vi_t0, true

    # Save predictions
    np.savez(
        os.path.join(PATH_DEST, f"preds_{MODEL_STR}_{FSTR_END_OUT}.npz"),
        y_pred = y_pred,
        y_true = y_true,
    )

    return

def lr_train(x_train, y_train):

    # TODO dynamic number of input channels and coefficients

    # Define number of input channels for gram matrix
    nin = 3 # complex wind (za), complex ice concentration (zci), complex constant 

    # Define number of complex coefficients
    nzm = 3 # A, B, C
   
    # Get dimensions for output arrays
    nt, _, nlat, nlon = np.shape(y_train)

    # Unpack target arrays
    ui_t0 = y_train[:,0,:,:]
    vi_t0 = y_train[:,1,:,:]

    # Unpack feature arrays
    ua_t0 = x_train[:,0,:,:]
    va_t0 = x_train[:,1,:,:]
    ci_t1 = x_train[:,2,:,:]

    # TODO fix nin for m_all (just a coincidence that it matches with nin)
    # need A, B, C (za, ua, constant)

    # TODO switch order of gram matrix so constant is at end?
    # for consisitency with lr equation Ax + Bx + C

    # Initialize output arrays
    ztrue_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # true present day ice velocity vector, complex
    zfit_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # present day fit ice velocity, complex
    zm_all = np.zeros((nzm, nlat, nlon), dtype = complex) # lr coefficients (A, B, C), complex
    
    # Iterate through each latitude, longitude gridpoint
    for ilat in range(nlat):
        for ilon in range(nlon):

            # Skip over land points
            if np.all(np.logical_or(np.isnan(ui_t0[:,ilat,ilon]), np.isnan(vi_t0[:,ilat,ilon]))):
                continue

            else:
                try:
                    # Handle missing data
                    
                    # Initialize mask for valid values
                    true_mask = np.ones_like(ui_t0[:,ilat,ilon], dtype=bool) # 1 = True = Inclusion

                    # Set 'True' for indices with nan values, 'False' for valid
                    inan = np.logical_or(np.isnan(ui_t0[:,ilat,ilon]), np.isnan(vi_t0[:,ilat,ilon]))

                    # Set NaN indices to False (Exclusion) (~ inverts 'True' where nan to 'False')
                    true_mask = ~inan

                    # Filter inputs to valid indices
                    ui_t0_filt = ui_t0[true_mask,ilat,ilon]
                    vi_t0_filt = vi_t0[true_mask,ilat,ilon]
                    ua_t0_filt = ua_t0[true_mask,ilat,ilon]
                    va_t0_filt = va_t0[true_mask,ilat,ilon]
                    ci_t1_filt = ci_t1[true_mask,ilat,ilon]

                    # Convert to complex
                    zi_t0 = ui_t0_filt + vi_t0_filt*1j # Complex 'today' ice velocity vector       
                    za_t0 = ua_t0_filt + va_t0_filt*1j # Complex 'today' wind vector
                    zci_t1 = ci_t1_filt + ci_t1_filt*1j # Complex 'yesterday' ice concentration
                    
                    # Store true complex ice velocity vectors at valid points
                    ztrue_all[true_mask, ilat, ilon] = zi_t0

                    # Define size of valid batch at current grid point
                    nt_ij = len(ua_t0_filt)

                    # Define gram matrix
                    G = np.ones(((nt_ij, n_in)), dtype = complex) 

                    G[:,0] = za_t0 # Present day wind velocity, complex
                    G[:,1] = zci_t1 # Previous day ice concentration, complex
                    
                    # NOTE last column of G constant

                    # Define data matrix
                    d = zi_t0.T

                    # Solve for lr coefficients
                    zm = (LA.inv((G.conj().T @ G))) @ G.conj().T @ d

                    # Save lr coefficients
                    for izm in range(len(zm)):
                        zm_all[izm, ilat, ilon] = m[izm]

                    # Calculate fit
                    zfit = G @ zm
                    
                    # Store predicted complex ice velocity vectors at valid points
                    zfit_all[true_mask, ilat, ilon] = zfit

                except Exception as e:
                    print(f"Error at lat={ilat}, lon={ilon}: {e}")

        print(f'lat {ilat} complete')
        
    return zm_all, zfit_all, ztrue_all

def lr_test(x_test, y_test, zm):

    # TODO dynamic number of input channels and coefficients

    # Define number of input channels for gram matrix
    nin = 3 # complex wind (za), complex ice concentration (zci), complex constant 
   
    # Get dimensions for output arrays
    nt, _, nlat, nlon = np.shape(y_test)

    # Unpack target arrays
    ui_t0 = y_test[:,0,:,:]
    vi_t0 = y_test[:,1,:,:]

    # Unpack feature arrays
    ua_t0 = x_test[:,0,:,:]
    va_t0 = x_test[:,1,:,:]
    ci_t1 = x_test[:,2,:,:]
      
    # Initialize output arrays
    ztrue_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # True complex 'today' ice velocity vectors
    zpred_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # Predicted complex 'today' ice velocity vectors
    
    # Iterate through each latitude, longitude gridpoint
    for ilat in range(nlat):
        for ilon in range(nlon):

            # Convert to complex
            zi_t0 = ui_t0[:,ilat,ilon] + vi_t0[:,ilat,ilon]*1j # Complex 'today' ice velocity vector       
            za_t0 = ua_t0[:,ilat,ilon] + va_t0[:,ilat,ilon]*1j # Complex 'today' wind vector
            zci_t1 = ci_t1[:,ilat,ilon] + ci_t1[:,ilat,ilon]*1j # Complex 'yesterday' ice concentration
            
            # Store true complex ice velocity vectors at valid points
            ztrue_all[:, ilat, ilon] = zi_t0

            # Define gram matrix
            G = np.ones(((nt, 3)), dtype = complex) # first column constant (1)

            G[:,1] = za_t0 # Complex wind, today
            G[:,2] = zci_t1 # Complex ice concentration, yesterday

            zm_ij = zm[:,ilat,ilon]

            # Calculate fit
            zpred = G @ zm_ij
            
            # Store predicted complex ice velocity vectors at valid points
            zpred_all[:, ilat, ilon] = zpred

        print(f'ilat {ilat} complete')
        
    return zpred_all, ztrue_all

