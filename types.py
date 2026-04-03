import numpy as np
import numpy.typing as npt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Standard types used throughout pipeline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Array3D = npt.NDArray[np.floating]  # (time, y, x)
Mask2D = npt.NDArray[np.bool_]      # (y, x)
Mask3D = npt.NDArray[np.bool_]      # (time, y, x)