"""Module returning a prediction function computed using epsilon-Support Vector Regression (SVR)
without slack variables (C=0).

Resolution of the problem is done using the cvxpy python third party library
(https://www.cvxpy.org/index.html) as toolbox for convex optimization.
NB: there is also a scikit-learn implementation of epsilon-SVR, but it does not
allow to set C=0 to perform "hard epsilon border".
"""

# Standard library imports
from typing import Callable

# Third party libraries imports
import cvxpy as cp
import numpy as np


def get_epsilon_sv_function(
    x_data: np.ndarray, y_data: np.ndarray, epsilon: float, verbose: bool = False
) -> tuple[Callable, dict]:
    """Compute the parameters of an epsilon-Support Vector Regression (SVR)
    prediction function for the dataset (x_data, y_data) with epsilon margin.
    Epsilon margin is "hard", i.e. no slack variables are used in the problem.
    If the dataset is not epsilon-SVR feasible, the problem will be infeasible.

    Parameters
    ----------
    x_data : np.ndarray
        Input data of shape (n_samples, n_features) or (n_samples,).
    y_data : np.ndarray
        Output data of shape (n_samples,) or (n_samples, 1).
    epsilon : float
        Epsilon margin for the SVR.
    verbose : bool, optional
        If True, the optimizer will be run in verbose mode and print
        information during the optimization process.

    Returns
    -------
    """

    # Reshape x_data if it's one-dimensional
    if x_data.ndim == 1:
        if y_data.ndim != 1:
            raise ValueError(
                "y_data must be one-dimensional if x_data is one-dimensional."
            )
        # Reshape x and y to be column vectors
        x_data = x_data.reshape(-1, 1)
    else:
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError("Number of samples in x_data and y_data must be the same.")
    if y_data.ndim != 1:
        if y_data.shape[1] != 1:
            raise ValueError("y_data must be one-dimensional or a column vector.")
    else:
        y_data = y_data.reshape(-1, 1)

    # Get number of samples and features from dataset
    print("x_line shape:", x_data.shape)
    print("y_line shape:", x_data.shape)
    _, p = x_data.shape  # _ = number of samples, p = number of features

    # Define and solve the CVXPY problem.
    omega = cp.Variable(shape=(p, 1), name="omega")
    b = cp.Variable(shape=1, name="b")
    cost = cp.sum_squares(omega)
    y_pred = b + x_data @ omega
    constraints = [
        y_data - y_pred <= epsilon,
        y_pred - y_data <= epsilon,
    ]
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # Solve problem and extract parameters of the prediction function
    prob.solve(verbose=verbose)

    def svr_function(x: np.ndarray) -> np.ndarray:
        """Epsilon-Support Vector Regression (SVR) prediction function.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
        np.ndarray
            Predicted output data of shape (n_samples,).
        """
        # Reshape x if it's one-dimensional
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.shape[1] != p:
            raise ValueError(f"Input data must have {p} features.")

        y_pred = b.value + x @ omega.value
        return y_pred.flatten()

    regression_params = {
        "epsilon": epsilon,
        "omega": omega.value,
        "b": b.value,
        "min_val": 0.5 * prob.value,
    }

    print(f"Regression function obtained is y= {omega.value}^T * x + {b.value}")

    return svr_function, regression_params
