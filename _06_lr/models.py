from . import lr_utils


class GridwiseClosedFormLR:
    def __init__(self, feature_fcn, target_fcn):
        self.feature_fcn = feature_fcn
        self.target_fcn = target_fcn
        # Initialize coefficients
        self.coef_ = None



        def fit(self, x, y):

            # Initialize complex parameters

            # Make complex features and targets
            za_t0, zci_t1, zi_t0 = 

            for i in range(self.width):
                for j in range(self.height):


            

    def fit(self, x, y):
        """
        
        """


    def test(self, x, parameters):
        """
        
        """


class GridwiseClosedFormWeightedLR():
    def __init__(self, in_channels, height, width):
        self.parameters_ = None

        