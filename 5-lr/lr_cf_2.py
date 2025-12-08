import numpy as np
from numpy import linalg as LA
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get global variables from master 'run-data-processing.sh'
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


def main():
    
    # Define input variable file name
    fnam = f'mask_norm_{HEM}{START_YEAR}{END_YEAR}_{TIMESTAMP_IN}.npz' # masked & normalized data  
    
    # Load input variables
    data = np.load(os.path.join(PATH_SOURCE, fnam))

    # Unpack input variables from .npz file
    uit = data['uitn'] # zonal ice velocity, present day, normalized
    vit = data['vitn'] # meridional ice velocity, present day, normalized
    uwt = data['uwtn'] # zonal wind velocity, present day, normalized
    vwt = data['vwtn'] # meridional wind velocity, present day, normalized
    icy = data['icyn'] # ice concentration, previous day, normalized

    # Pack input variables into list
    invars = [uit, vit, uwt, vwt, icy]

    print("Input Variables Loaded")



def lr_train(invars):
   
    # Get ice velocity for nan filtering
    uit = invars[0]
    vit = invars[1]

    # Get dimensions for output arrays
    nt, ny, nx = np.shape(invars[0])
      
    # Initialize output arrays
    true_all = np.full((nt, ny, nx), np.nan, dtype = complex) # True complex 'today' ice velocity vectors
    fit_all = np.full((nt, ny, nx), np.nan, dtype = complex) # Predicted complex 'today' ice velocity vectors
    m_all = np.zeros((3, ny, nx), dtype = complex) # Complex model parameters (mean, complex 'yesterday' wind, complex 'yesterday' concentration)
    
    # Iterate through each latitude, longitude gridpoint
    for iy in range(ny):
        for ix in range(nx):

            # Skip over land points
            if np.all(np.logical_or(np.isnan(uit[:,iy,ix]), np.isnan(vit[:,iy,ix]))):
                continue

            else:
                try:
                    # Handle missing data
                    
                    # Initialize mask for valid values
                    true_mask = np.ones_like(uit[:,iy,ix], dtype=bool) # 1 = True = Inclusion

                    # Set 'True' for indices with nan values, 'False' for valid
                    inan = np.logical_or(np.isnan(uit[:,iy,ix]), np.isnan(vit[:,iy,ix]))

                    # Set NaN indices to False (Exclusion) (~ inverts 'True' where nan to 'False')
                    true_mask = ~inan

                    # Filter inputs to valid indices and unpack input list
                    uit_f, vit_f, uwt_f, vwt_f, icy_f = [var[true_mask,iy,ix] for var in invars]

                    # Convert to complex
                    it_c = uit_f + vit_f*1j # Complex 'today' ice velocity vector       
                    wt_c = uwt_f + vwt_f*1j # Complex 'today' wind vector
                    icy_c = icy_f + icy_f*1j # Complex 'yesterday' ice concentration
                    
                    # Store true complex ice velocity vectors at valid points
                    true_all[true_mask, iy, ix] = it_c

                    # Define gram matrix
                    G = np.ones(((len(it_c), 3)), dtype = complex) # first column constant (1)

                    G[:,1] = wt_c # Complex wind, today
                    G[:,2] = icy_c # Complex ice concentration, yesterday

                    # Define data matrix
                    d = it_c.T

                    # Solve for model parameters
                    m = (LA.inv((G.conj().T @ G))) @ G.conj().T @ d

                    # Save model parameters
                    for i in range(len(m)):
                        m_all[i, iy, ix] = m[i]

                    # Calculate fit
                    fit = G @ m
                    
                    # Store predicted complex ice velocity vectors at valid points
                    fit_all[true_mask, iy, ix] = fit

                except Exception as e:
                    print(f"Error at iy={iy}, ix={ix}: {e}")


        print(f'iy {iy} complete')
        
    return m_all, fit_all, true_all

def lr_test(invars, m):

    # Get ice velocity for nan filtering
    uit = invars[0]
    vit = invars[1]

    # Get dimensions for output arrays
    nt, ny, nx = np.shape(invars[0])
      
    # Initialize output arrays
    true_all = np.full((nt, ny, nx), np.nan, dtype = complex) # True complex 'today' ice velocity vectors
    pred_all = np.full((nt, ny, nx), np.nan, dtype = complex) # Predicted complex 'today' ice velocity vectors
    
    # Iterate through each latitude, longitude gridpoint
    for iy in range(ny):
        for ix in range(nx):

            # Filter inputs to valid indices
            uit_f, vit_f, uwt_f, vwt_f, icy_f = [var[:,iy,ix] for var in invars]

            # Convert to complex
            it_c = uit_f + vit_f*1j # Complex 'today' ice velocity vector       
            wt_c = uwt_f + vwt_f*1j # Complex 'today' wind vector
            icy_c = icy_f + 0*1j # Complex 'yesterday' ice concentration
            
            # Store true complex ice velocity vectors at valid points
            true_all[:, iy, ix] = it_c

            # Define gram matrix
            G = np.ones(((len(it_c), 3)), dtype = complex) # first column constant (1)

            G[:,1] = wt_c # Complex wind, today
            G[:,2] = icy_c # Complex ice concentration, yesterday

            m_grid = m[:,iy,ix]

            # Calculate fit
            pred = G @ m_grid
            
            # Store predicted complex ice velocity vectors at valid points
            pred_all[:, iy, ix] = pred

        print(f'iy {iy} complete')
        
    return pred_all, true_all

