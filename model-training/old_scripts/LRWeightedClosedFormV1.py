import numpy as np
from numpy import linalg as LA
import os
from scipy.sparse import diags

PATH_SOURCE = "/home/jbassham/jack/data/w92_20/inputs"
PATH_DEST = "/home/jbassham/jack/data/w92_20/outputs"

EPSILON = 0.001

def lr_weighted_gridwise(invars, epsilon):
    
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

                    # Filter inputs to valid indices
                    uit_f, vit_f, rt_f, uwt_f, vwt_f, icy_f = [var[true_mask,iy,ix] for var in invars]

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

                    # Complex uncertainties
                    rr = 2 * ((rt_f ** 2) + epsilon)
                    rr = 1 / rr

                    # Create diagonal matrix of uncertainty (scipy sparse for better memory)
                    R = diags(rr)

                    # Solve for weighted model parameters
                    m = (LA.inv((G.conj().T @ R @ G))) @ G.conj().T @ R @ d # (adapted from eqn 39, SIOC221B Lec 10)

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

def main():

    # Load input variable file
    fnam = 'inputs_normalized.npz'
    data = np.load(os.path.join(PATH_SOURCE, fnam))

    # Unpack input variables from .npz file
    uit = data['uitn']
    vit = data['vitn']
    rt = data['rtn']
    uwt = data['uwtn']
    vwt = data['vwtn']
    icy = data['icyn']

    # Pack input variables into list
    invars = [uit, vit, rt, uwt, vwt, icy]

    print("Input Variables Loaded")

    # Run regression
    m, fit, true = lr_weighted_gridwise(invars, EPSILON)

    print("All Points Complete")

    # Save output
    fnam = 'lr_weighted_gridwise_v1.npz'

    np.savez(os.path.join(PATH_DEST, fnam), m = m, fit = fit, true = true)

    print(f"LR outputs saved at: \n {PATH_DEST}/{fnam}")

    return

if __name__ == "__main__":
    main()
