import numpy as np
from tqdm import tqdm

class BaseGridwiseLR:
    def __init__(self, feature_fcn, target_fcn):
        self.feature_fcn = feature_fcn
        self.target_fcn = target_fcn
        # Initialize coefficients
        self.coef_ = None

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

        # Initialize coefficients
        self.coef_ = np.full((in_features, height, width), np.nan, dtype=complex)

        # Loop through gridpoints
        for j in range(height):
            for i in tqdm(range(width), desc='Gridpoints', leave=False):

                X_ji = X[:,:,j,i]
                y_ji = y[:,:,j,i]

            # Get weights if passed to model instance
            w_ji = None if w is None else w[:,j,i]

            try:
                # Try to solve for coefficients
                self.coef_[:,j,i] = self._solve(X_ji, y_ji, w_ji)
            
            # Print any errors that occur in solving at a gridpoint
            except Exception as e:
                print(f'Failed at (j={j}, i={i}): {e}')

        return self
    

    def predict(self, x):
        """
        
        """

        # Check that model was fit and coefficients exist before predicting
        if self.coef_ is None:
            raise ValueError('Fit model to solve for coefficients before making predictions')
        
        




class GridwiseClosedFormWeightedLR():
    def __init__(self, in_channels, height, width):
        self.parameters_ = None


        # Solve for coefficients at gridpoint
        self.coef_[:,j,i] = (np.linalg.inv((X_ji.conj().T @ X_ji))) @ X_ji.conj.T @ y_ji.T
        