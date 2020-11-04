import numpy as np
import torch


def fine_tune_classification(model, val):
    """Wraps the classification task for optimization purposes.

    Args:
        model (DRBM): Child object from DRBM class.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(w):
        """Gathers weights from the meta-heuristic and evaluates over validation data.

        Args:
            w (float): Array of variables.

        Returns:
            1 - accuracy of validation set.

        """

        # Reshaping optimization variables to appropriate size
        W_cur = np.reshape(w, (model.W.size(0), model.W.size(1)))

        # Converting numpy to tensor
        W_cur = torch.from_numpy(W_cur).float()

        # Replaces the model's weight
        model.W = torch.nn.Parameter(W_cur)

        # Classifies over validation set
        acc, _, _ = model.predict(val)

        return 1 - acc.item()

    return f


def fine_tune_reconstruction(model, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        model (RBM): Child object from RBM class.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(w):
        """Gathers weights from the meta-heuristic and evaluates over validation data.

        Args:
            w (float): Array of variables.

        Returns:
            Mean squared error (MSE) of validation set.

        """

        # Reshaping optimization variables to appropriate size
        W_cur = np.reshape(w, (model.W.size(0), model.W.size(1)))

        # Converting numpy to tensor
        W_cur = torch.from_numpy(W_cur).float()

        # Replaces the model's weight
        model.W = torch.nn.Parameter(W_cur)

        # Reconstructs over validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
