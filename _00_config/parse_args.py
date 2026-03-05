import argparse


def parse_args():

    # Assign instance of argument parser
    parser = argparse.ArgumentParser()

    # Define data step timestamp version arguments for parser
    parser.add_argument('--timestamp_out')
    parser.add_argument('--timestamp_raw')
    parser.add_argument('--timestamp_regrid')
    parser.add_argument('--timestamp_coordinates')
    parser.add_argument('--timestamp_mask_norm')
    parser.add_argument('--timestamp_model_inputs')
    parser.add_argument('--timestamp_model_output')

    # Return the arguments
    return parser.parse_args()