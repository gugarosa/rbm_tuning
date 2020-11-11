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
    parser = argparse.ArgumentParser(usage='Optimizes an RBM model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist'])

    parser.add_argument('model_name', help='Model identifier', choices=['drbm', 'rbm'])

    parser.add_argument('parameter', help='Parameter identifier', choices=['a', 'b', 'c', 'U', 'W'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['pso'])

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-bounds', help='Searching bounds', type=float, default=0.01)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=15)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    name = args.model_name
    parameter = args.parameter
    seed = args.seed
    batch_size = args.batch_size

    # Gathering optimization variables
    bounds = args.bounds
    n_agents = args.n_agents
    n_iterations = args.n_iter
    mh_name = args.mh
    mh = o.get_mh(mh_name).obj
    hyperparams = o.get_mh(args.mh).hyperparams

    # Loads the data
    _, val, _ = l.load_dataset(name=dataset, seed=seed)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads pre-trained model
    model = torch.load(f'{name}.pth')

    # Gathering parameter from desired layer and expanding its dimensions
    # Note this makes sure that one-dimensional variables work
    param = getattr(model, parameter).detach().cpu().numpy()
    param = np.expand_dims(param, -1)

    # Defining lower and upper bounds, and number of variables
    lb = list(np.reshape(param - bounds, param.shape[0] * param.shape[1]))
    ub = list(np.reshape(param + bounds, param.shape[0] * param.shape[1]))
    n_variables = param.shape[0] * param.shape[1]

    # Checks the optimization task
    if name == 'rbm':
        # Defines accordingly
        opt_fn = t.fine_tune_reconstruction(model, parameter, val)

    elif name == 'drbm':
        # Defines accordingly
        opt_fn = t.fine_tune_classification(model, parameter, val)
    
    # Defines the name of output file
    opt_name = f'{mh_name}_{name}'

    # Running the optimization task
    history = opt.optimize(mh, opt_fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)

    # Saving history object
    history.save(f'{opt_name}.history')

    # Reshaping parameter to appropriate size
    best_param = np.reshape(history.best_agent[-1][0], (param.shape[0], param.shape[1]))

    # Converting numpy to tensor and squeeze its last dimension
    # Note this makes sure that one-dimensional variables work
    best_param = torch.from_numpy(best_param).float()
    best_param = torch.squeeze(best_param, -1)

    # Replaces the model's parameter
    setattr(model, parameter, torch.nn.Parameter(best_param))

    # Saving optimized model
    torch.save(model, f'{opt_name}.optimized')
