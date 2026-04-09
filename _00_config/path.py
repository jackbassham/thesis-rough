import os
from .config import(
    HEM, 
    TIMESTAMP_RAW,
    TIMESTAMP_REGRID,
    TIMESTAMP_COORDINATES,
    TIMESTAMP_NAN_MASK,
    TIMESTAMP_R,
    TIMESTAMP_MASK_NORM,
    TIMESTAMP_INPUTS,
    TIMESTAMP_OUTPUTS,
)

# Define path root to data scratch directory
ROOT_DATA = '/data/globus/jbassham/thesis-rough'

# Define path root to project (one step above module)
ROOT_PROJECT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
    )
)


def build_data_path(data_directory: str, data_timestamp: str, root: str = ROOT_DATA) -> str:
    """

    """
    data_path = (
        os.path.join(
            root,
            data_directory,
            HEM, 
            data_timestamp,
        )
    )

    return data_path


def build_model_output_path(model_name: str, model_timestamp: str = TIMESTAMP_OUTPUTS, root: str = ROOT_DATA) -> str:
    """

    """
    model_output_path = (
        os.path.join(
            root,
            'model-output',
            model_name,
            HEM,
            model_timestamp,
        )
    )

    return model_output_path


def build_plot_path(model_name: str, model_timestamp: str = TIMESTAMP_OUTPUTS, root: str = ROOT_PROJECT) -> str:
    """

    """
    plot_path = (
        os.path.join(
            root,
            'plots',
            model_name,
            HEM,
            model_timestamp,
        )
    )

    return plot_path


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store possible data directory paths as global variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH_RAW = build_data_path('raw', TIMESTAMP_RAW)
PATH_REGRID = build_data_path('regrid', TIMESTAMP_REGRID)
PATH_COORDINATES = build_data_path('coordinates', TIMESTAMP_COORDINATES)
PATH_NAN_MASK = build_data_path('mask-norm', TIMESTAMP_NAN_MASK)
PATH_R = build_data_path('model-input', TIMESTAMP_R)
PATH_MASK_NORM = build_data_path('mask-norm', TIMESTAMP_MASK_NORM)
PATH_MODEL_INPUTS = build_data_path('model-inputs', TIMESTAMP_INPUTS)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store possible model output paths as global variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH_PS_OUT = build_model_output_path('ps')
PATH_LR_CF_OUT = build_model_output_path('lr-cf')
PATH_LR_WTD_CF_OUT = build_model_output_path('lr-wtd-cf')
PATH_CNN_PT_OUT = build_model_output_path('cnn')
PATH_CNN_WTD_PT_OUT = build_model_output_path('cnn-wtd')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store possible output plot paths as global variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH_PS_OUT = build_plot_path('ps')
PATH_LR_CF_OUT = build_plot_path('lr_cf')
PATH_LR_WTD_CF_OUT = build_plot_path('lr_wtd_cf')
PATH_CNN_PT_OUT = build_plot_path('cnn')
PATH_CNN_WTD_PT_OUT = build_plot_path('cnn_wtd')