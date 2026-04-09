from _00_config.load_config import load_config
from _00_config.parse_args import parse_args

from typing import Callable
# TODO Import PipelineConfig for type checking only??

# FIXME parse_args import into both load_config and run_pipeline

def main():

    # TODO make updates look at readme

    # Define pipeline steps in dict with argument keys and callable values
    pipeline_steps = {
        'download_ice_vel': step_download_ice_vel,
        'download_ice_conc': step_download_ice_conc,
        'download_wind': step_download_wind,
        'regrid_ice_vel': step_regrid_ice_vel,
        'regrid_ice_conc': step_regrid_ice_conc,
        'regrid_wind': step_regrid_wind,
        'make_coordinates': step_make_coordinates,
        'mask_normalize': step_mask_normalize,
        'process_inputs': step_process_inputs,
        # 'ps': step_ps,
        # 'lr': step_lr,
        # 'lr_wtd': step_lr_wtd,
        'cnn': step_cnn,
        # 'cnn_wtd': step_cnn_wtd,
    }

    # Instantiate argument parser
    args = parse_args()

    # Instantiate configuration from loader
    pipeline_config = load_config()

    # Run pipeline with configuration instance and optional start stop points
    run_pipeline(
        pipeline_steps,
        pipeline_config,
        start = args.start,
        stop = args.stop,
    )


def run_pipeline(pipeline_steps: dict[str, Callable], pipeline_config, 
                 start: str | None = None, stop: str | None = None):
    """
    
    """

    # Get pipeline step keys from the dict and store in list
    steps = list(pipeline_steps.keys())

    # Initialize start index
    start_index = 0

    # Initialize stop index
    stop_index = len(pipeline_steps)

    # Move start index if command line argument provided by user
    if start:
        # Handle case where start is not valid
        if start not in steps:
            raise ValueError(f'Unknown pipeline start step: "{start}"')

        else:
            # Retrieve index from start step
            start_index = steps.index(start)

    # Move stop index if command line argument provided by user
    if stop:
        if stop not in steps:
            raise ValueError(f'Unknown pipeline stop step: "{stop}"')
        
        else:
            # Retrieve index from stop step (exclusive of stop, add 1)
            stop_index = steps.index(stop) + 1

    # Iterate through pipeline steps given indices
    for step in steps[start_index:stop_index]:

        print(f'Running pipeline step: {step}')

        # Get function for pipeline step and run
        pipeline_steps[step](pipeline_config)

        print(f'Completed pipeline step: {step}')
        print(' ')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper Functions for Pipeline Steps
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO reduce repetition, add error handling, allow for different datasets?

# TODO remove number ordering for modules to allow flexibility?

# TODO quick eval in each model step


def step_download_ice_vel(config):
    from _01_access_data.download_ice_vel import main
    main(config)


def step_download_ice_conc(config):
    from _01_access_data.download_ice_conc import main
    main(config)


def step_download_wind(config):
    from _01_access_data.download_wind_jra55 import main
    main(config)


def step_regrid_ice_vel(config):
    from _02_regrid.regrid_ice_vel import main
    main(config)


def step_regrid_ice_conc(config):
    from _02_regrid.regrid_ice_conc import main
    main(config)


def step_regrid_wind(config):
    from _02_regrid.regrid_wind import main
    main(config)

def step_make_coordinates(config):
    from _02_regrid.make_coordinates import main
    main(config)


def step_mask_normalize(config):
    from _03_mask_normalize.mask_normalize import main 
    main(config)


def step_process_inputs(config):
    from _04_process_inputs.make_inputs import main
    main(config)


# def step_ps(config):
#     from _05_ps.ps import main
#     main(config)


# def step_lr(config):
#     from _06_lr.lr_cf import main
#     main(config)


# def step_lr_wtd(config):
#     from _07_lr_weighted.lr_wtd_cf import main
#     main(config)


def step_cnn(config):
    from _08_cnn.cnn_pt import main
    main(config)

    from _10_evaluate.quick_eval import run_eval
    # Run quick eval for the cnn-pt
    run_eval(config, 'cnn_pt')


# def step_cnn_wtd(config):
#     from _09_cnn_weighted.cnn_wtd_pt import main
#     main(config)


if __name__ == '__main__':
    main()

