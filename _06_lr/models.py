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
        n_samples, _, height, width = x.shape

        # Build feature matrix (n_samples, n_features+1, height, width)
        X = self.feature_fcn(x)

        # Build target vector
        y = self.target_fcn(y)

        # Infer number of features from feature matrix
        in_features = X.shape[1]

        # Initialize coefficients
        self.coef_ = np.full((in_features, height, width), np.nan, dtype=complex)

        # Loop through gridpoints
        for i in range(width):
            for j in range(height):

            X_ji = X[:, :, j, i]
            y_ji = y[:, :, j, i]

            # Solve for coefficients at gridpoint



            

    def fit(self, x, y):
        """
        
        """


    def test(self, x, parameters):
        """
        
        """


class GridwiseClosedFormWeightedLR():
    def __init__(self, in_channels, height, width):
        self.parameters_ = None

        