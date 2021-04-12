import argparse
import glob

import numpy as np
import opytimizer.visualization.convergence as c
from natsort import natsorted
from opytimizer.utils.history import History


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Analyzes an RBM optimization convergence.')

    parser.add_argument('input_files', help='General input history file (without seed)', type=str, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    input_files = args.input_files

    # Instantiates a list of overall means
    input_files_means = []

    # Iterates through all input files
    for i, input_file in enumerate(input_files):
        # Checks the folder and creates a list of model files
        input_file_with_seeds = natsorted(glob.glob(f'{input_file}*.history'))

        # Instantiates a list of seeded files means
        input_file_with_seeds_values = []

        # Iterates over every seeded file
        for input_file_with_seed in input_file_with_seeds:
            # Instantiates and loads a History object 
            h = History()
            h.load(input_file_with_seed)

            # Gathers the bets agent's fitness and appends to seeded list
            best_agent_pos = h.get(key='best_agent', index=(1,))
            input_file_with_seeds_values.append(best_agent_pos)

        # Calculates the mean and appends to overall
        mean = np.mean(input_file_with_seeds_values, axis=0)
        input_files_means.append(mean)

    # Plots the convergence of best fitnesses
    c.plot(*input_files_means)
