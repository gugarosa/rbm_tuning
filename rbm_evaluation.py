import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.loader as l
import utils.objects as o
import utils.optimizer as opt
import utils.target as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates a pre-trained RBM model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist'])

    parser.add_argument('input_file', help='Input model file identifier', type=str)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    input_file = args.input_file
    seed = args.seed

    # Loads the data
    _, _, test = l.load_dataset(name=dataset, seed=seed)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads pre-trained model
    model = torch.load(input_file)

    # Checks if model has an prediction function
    if hasattr(model, 'predict'):
        # Predicts the test set
        output, _, _ = model.predict(test)

    # If there is no prediction
    else:
        # Reconstructs the test set
        output, _ = model.reconstruct(test)

    print(output)
