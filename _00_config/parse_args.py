import argparse


def parse_args():

    # Assign instance of argument parser
    parser = argparse.ArgumentParser()

    # Define data stage timestamp command line arguments 
    parser.add_argument('--timestamp_out')
    parser.add_argument('--timestamp_raw')
    parser.add_argument('--timestamp_regrid')
    parser.add_argument('--timestamp_coordinates')
    parser.add_argument('--timestamp_mask_norm')
    parser.add_argument('--timestamp_model_inputs')
    parser.add_argument('--timestamp_model_output')

    # Define command line argument for pipeline start point (optional partial run)
    parser.add_argument(
        '--start',
        type = str,
        default = None,
        help = 'Pipeline step to start from, runs from start point on'
    )

    # Define command line argument for pipeline stop point (optional partial run)
    parser.add_argument(
        '--stop',
        type = str,
        default = 'None',
        help = 'Pipeline step to end on, runs up until stop point'
    )

    # Return the arguments
    return parser.parse_args()