import numpy as np
from tqdm import tqdm
from . import lr_utils
import scipy

class BaseGridwiseLR:
    def __init__(self, feature_fcn, target_fcn):
        self.feature_fcn = feature_fcn
        self.target_fcn = target_fcn
        # Initialize complex coefficients
        self.Z_coef_ = None


    def _solve(self, X, y, w=None):
        """
        Overide in subclasses
        """
        raise NotImplementedError
    

    def fit(self, x, y, r=None):
        """
        
        """

        # Infer shape from inputs
        _, _, height, width = x.shape

        # Build feature matrix (n_samples, n_features, height, width)
        X = self.feature_fcn(x)

        # Build target vector
        y = self.target_fcn(y)

        # Get weights if uncertainty passed to model instance
        w = None
        if r is not None:
            w = self._buid_weights(r)

        # Infer number of features from feature matrix
        in_features = X.shape[1]

        # Initialize complex coefficients array
        self.Z_coef_ = np.full((in_features, height, width), np.nan, dtype=complex)

        # Loop through gridpoints
        for j in range(height):
            for i in tqdm(range(width), desc='Gridpoints', leave=False):

                X_ji = X[:,:,j,i]
                y_ji = y[:,:,j,i]

            # Get weights if passed to model instance
            w_ji = None if w is None else w[:,j,i]

            try:
                # Try to solve for complex coefficients
                self.Z_coef_[:,j,i] = self._solve(X_ji, y_ji, w_ji)
            
            # Print any errors that occur in solving at a gridpoint
            except Exception as e:
                print(f'Failed at (j={j}, i={i}): {e}')

        return self
    

    def R_coef_(self):
        """
        
        """

        # Initialize real coefficient array with shape inferred by complex array
        # (n_Re * n_Im, height, width)
        m = np.empty((2*self.Z_coef_.shape[0], *self.Z_coef_.shape[1:]))

        # Starting at the first entry, every other entry is Real
        m[0::2] = self.Z_coef_.real

        # Starting at the second entry, every other entry is Imaginary
        m[1::2] = self.Z_coef_.imag

        return m
    

    def predict(self, x):
        """
        
        """

        # Check that model was fit and complex coefficients exist before predicting
        if self.Z_coef_ is None:
            raise ValueError('Fit model to solve for coefficients before making predictions')

        # Infer shape from inputs
        n_samples, _, height, width = x.shape

        # Initialize complex predictions array
        Z_preds_ = np.full((n_samples, height, width), np.nan, dtype=complex)

        # Build feature matrix (n_samples, n_features, height, width)
        X = self.feature_fcn(x)

        # Loop through gridpoints
        for j in range(height):
            for i in tqdm(range(width), desc='Gridpoints', leave=False):

                X_ji = X[:,:,j,i]
                # Use coefficients to make prediction
                Z_preds_[:,j,i] = X_ji @ self.Z_coef_[:,j,i]

        return Z_preds_
    

    def _z_to_vector(z):
        """

        """

        return np.stack(
            [z.real, z.imag],
            axis=1
        )
    
    
    def R_preds_(self):
        return self._z_to_vector(self.Z_preds_)


class UnweightedLR(BaseGridwiseLR):
    # Overide base solve with closed form solution to LR
    def _solve(self, X, y, w=None):
        return np.linalg.inv((X.conj().T @ X)) @ X.conj.T @ y.T


class WeightedLR(BaseGridwiseLR):
    def __init__(self, feature_fcn, target_fcn, weight_fcn):
        super().__init__(feature_fcn, target_fcn)
        self.weight_fcn = weight_fcn

    def _build_weights(self, r):
        # Build complex weights from uncertainty
        return self.weight_fcn(r)
    
    # Overide base solve with closed form solution to Weighted LR
    def _solve(self, X, y, w):

        # Diagonalize weights using sparse diags
        W = scipy.sparse.diags(w)

        # Solve with close form solution
        return np.linalg.inv(X.conj().T @ W @ X) @ X.conj().T @ W @ y



        