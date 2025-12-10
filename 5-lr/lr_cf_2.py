import numpy as np
from numpy import linalg as LA
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'train-models-all.sh'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HEM = os.getenv("HEM") # Hemisphere (sh or nh)

START_YEAR = int(os.getenv("START_YEAR")) # data starts 01JAN<START_YEAR>
END_YEAR = int(os.getenv("END_YEAR")) # data ends 31DEC<END_YEAR>

TIMESTAMP_IN = os.getenv("TIMESTAMP_IN") # timestamp version of input data

TIMESTAMP_MODEL = os.getenv("TIMESTAMP_MODEL") # timestamp version of the model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths to data directories defined here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Define masked & normalized input data path; relative to current
PATH_SOURCE = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        'mask-norm', 
        HEM,
        TIMESTAMP_IN)
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_SOURCE, exist_ok=True)

# Define model output data input path; relative to current
PATH_DEST = os.path.abspath(
    os.path.join(
        script_dir, 
        '..', 
        'data', 
        'model-output',
        'lr', 
        HEM,
        TIMESTAMP_MODEL)
)

# Create the directory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additional global variables here
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSTR_END_IN = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}"
FSTR_END_MODEL = f"{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_MODEL}"

def main():
    
    # Load in training data
    data = np.load(os.path.join(PATH_SOURCE,f'train_{FSTR_END_IN}.npz'))
    x_train = data['x_train']
    y_train = data['y_train']

    # Load in testing data
    data = np.load(os.path.join(PATH_SOURCE,f'test_{FSTR_END_IN}.npz'))
    x_test = data['x_test']
    y_test = data['y_test']

    # Train model
    m, fit_train, true_train = lr_train(x_train, y_train)

    # Get predictions on test set
    pred_test, true_test = lr_test(x_test, y_test, m)

    # Save coeffients, fit
    np.savez(
        os.path.join(PATH_DEST, f"lr_coeff_fit_{FSTR_END_MODEL}"),
        m = m,
        fit_train = fit_train,
        true_train = true_train
    )

    # Save predictions
    np.savez(
        os.path.join(PATH_DEST, f"lr_preds_{FSTR_END_MODEL}"),
        pred_test = pred_test,
        true_test = true_test
    )

    return

def lr_train(x_train, y_train):

    # Get number of input channels for gram matrix
    _, nin, _, _ = np.shape(x_train)
   
    # Get dimensions for output arrays
    nt, _, nlat, nlon = np.shape(y_train)

    # Unpack target arrays
    ui_t0 = y_train[:,0,:,:]
    vi_t0 = y_train[:,1,:,:]

    # Unpack feature arrays
    ua_t0 = x_train[:,0,:,:]
    va_t0 = x_train[:,1,:,:]
    ci_t1 = x_train[:,2,:,:]
    
    # Initialize output arrays
    true_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # true present day ice velocity vector, complex
    fit_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # present day fit ice velocity, complex
    m_all = np.zeros((nin, nlat, nlon), dtype = complex) # lr coefficients (mean, present day wind, present day concentration), complex
    
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
                    true_all[true_mask, ilat, ilon] = zi_t0

                    # Define gram matrix
                    G = np.ones(((len(ua_t0), 3)), dtype = complex) # first column constant (1)

                    G[:,1] = za_t0 # Complex wind, today
                    G[:,2] = zci_t1 # Complex ice concentration, yesterday

                    # Define data matrix
                    d = zi_t0.T

                    # Solve for lr coefficients
                    m = (LA.inv((G.conj().T @ G))) @ G.conj().T @ d

                    # Save lr coefficients
                    for im in range(len(m)):
                        m_all[im, ilat, ilon] = m[im]

                    # Calculate fit
                    fit = G @ m
                    
                    # Store predicted complex ice velocity vectors at valid points
                    fit_all[true_mask, ilat, ilon] = fit

                except Exception as e:
                    print(f"Error at lat={ilat}, lon={ilon}: {e}")

        print(f'lat {ilat} complete')
        
    return m_all, fit_all, true_all

def lr_test(x_test, y_test, m):

    # Get number of input channels for gram matrix
    _, nin, _, _ = np.shape(x_test)
   
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
    true_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # True complex 'today' ice velocity vectors
    pred_all = np.full((nt, nlat, nlon), np.nan, dtype = complex) # Predicted complex 'today' ice velocity vectors
    
    # Iterate through each latitude, longitude gridpoint
    for ilat in range(nlat):
        for ilon in range(nlon):

            # Convert to complex
            zi_t0 = ui_t0[:,ilat,ilon] + vi_t0[:,ilat,ilon]*1j # Complex 'today' ice velocity vector       
            za_t0 = ua_t0[:,ilat,ilon] + va_t0[:,ilat,ilon]*1j # Complex 'today' wind vector
            zci_t1 = ci_t1[:,ilat,ilon] + ci_t1[:,ilat,ilon]*1j # Complex 'yesterday' ice concentration
            
            # Store true complex ice velocity vectors at valid points
            true_all[:, ilat, ilon] = zi_t0

            # Define gram matrix
            G = np.ones(((len(ua_t0), 3)), dtype = complex) # first column constant (1)

            G[:,1] = za_t0 # Complex wind, today
            G[:,2] = zci_t1 # Complex ice concentration, yesterday

            m_ij = m[:,ilat,ilon]

            # Calculate fit
            pred = G @ m_ij
            
            # Store predicted complex ice velocity vectors at valid points
            pred_all[:, ilat, ilon] = pred

        print(f'ilat {ilat} complete')
        
    return pred_all, true_all

