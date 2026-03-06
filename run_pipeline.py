from _00_config.load_config import load_config
from _00_config.parse_args import parse_args

# Define pipeline steps
PIPELINE_STEPS = {
    'download_motion': step_download_motion,
    'download_concentration': step_download_concentration,
    'download_wind': step_download_wind,
    'regrid_motion': step_regrid_motion,
    'regrid_concentration': step_regrid_concentration,
    'regrid_wind': step_regrid_wind,
    'mask_normalize': step_mask_normalize,
    'process_inputs': step_process_inputs,
    'ps': step_ps,
    'lr': step_lr,
    'lr_wtd': step_lr_wtd,
    'cnn': step_cnn,
    'cnn_wtd': step_cnn_wtd,
}


def main():

    # Instantiate configuration
    config = load_config()


def run_pipeline(config, start = None, end = None):
    """
    
    """

    # Get pipeline step keys from the dict and store in list
    steps = list(PIPELINE_STEPS.keys())

    # Initialize start index
    start_index = 0

    # Initialize end index
    end_index = len(steps)

    # Move start index if command line argument provided by user
    if start:
        start_index = PIPELINE_STEPS.index(start)

    # Move end index if command line argument provided by user
    if end:
        # Slice exclusive of end, so add one to index
        end_index = PIPELINE_STEPS.index(end) + 1

    # Iterate through pipeline steps given indices
    for step in steps[start_index:end_index]:

        print(f'Running pipeline step: {step}')

        # Get function for pipeline step and run
        PIPELINE_STEPS[step](config)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pipeline Step Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO reduce repetition, add error handling, allow for different datasets?

# TODO remove number ordering for modules to allow flexibility?

# TODO quick eval in each model step


def step_download_motion(config):
    from ._01_download.download_motion_pp import main
    main(config)

def step_download_concentration(config):
    from ._01_download.download_concentration_nimbus7 import main
    main(config)


def step_download_wind(config):
    from ._01_download.download_wind_jra55 import main
    main(config)


def step_regrid_motion(config):
    from ._02_regrid.regrid_motion_pp import main
    main(config)


def step_regrid_concentration(config):
    from ._02_regrid.regrid_concentration_nimbus7 import main
    main(config)


def step_regrid_wind(config):
    from ._02_regrid.regrid_wind_jra55 import main
    main(config)


def step_mask_normalize(config):
    from ._03_mask_normalize.mask_normalize import main 
    main(config)


def step_process_inputs(config):
    from ._04_process_inputs.make_inputs import main
    main(config)


def step_ps(config):
    from ._05_ps.ps import main
    main(config)


def step_lr(config):
    from ._06_lr.lr_cf import main
    main(config)


def step_lr_wtd(config):
    from ._07_lr_weighted.lr_wtd_cf import main
    main(config)


def step_cnn(config):
    from ._08_cnn.cnn_pt import main
    main(config)


def step_cnn_wtd(config):
    from ._09_cnn_weighted.cnn_wtd_pt import main
    main(config)


if __name__ == '__main__':
    main()

