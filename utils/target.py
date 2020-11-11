import numpy as np
import torch


def fine_tune_classification(model, parameter, val):
    """Wraps the classification task for optimization purposes.

    Args:
        model (DRBM): Child object from DRBM class.
        parameter (torch.Tensor): Parameter to be optimized.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(w):
        """Gathers weights from the meta-heuristic and evaluates over validation data.

        Args:
            w (float): Array of variables.

        Returns:
            1 - accuracy of validation set.

        """

        # Gathering shape of parameter
        param = getattr(model, parameter).detach().cpu().numpy()
        param = np.expand_dims(param, -1)

        # Reshaping optimization variable to appropriate size
        current_param = np.reshape(w, (param.shape[0], param.shape[1]))

        # Converting numpy to tensor
        current_param = torch.from_numpy(current_param).float()

        # Replaces the model's parameter
        setattr(model, parameter, torch.nn.Parameter(current_param))

        # Classifies over validation set
        acc, _, _ = model.predict(val)

        return 1 - acc.item()

    return f


def fine_tune_reconstruction(model, parameter, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        model (RBM): Child object from RBM class.
        parameter (torch.Tensor): Parameter to be optimized.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(w):
        """Gathers weights from the meta-heuristic and evaluates over validation data.

        Args:
            w (float): Array of variables.

        Returns:
            Mean squared error (MSE) of validation set.

        """

        # Gathering shape of parameter
        param = getattr(model, parameter).detach().cpu().numpy()
        param = np.expand_dims(param, -1)

        # Reshaping optimization variable to appropriate size
        current_param = np.reshape(w, (param.shape[0], param.shape[1]))

        # Converting numpy to tensor
        current_param = torch.from_numpy(current_param).float()

        # Replaces the model's parameter
        setattr(model, parameter, torch.nn.Parameter(current_param))

        # Reconstructs over validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
