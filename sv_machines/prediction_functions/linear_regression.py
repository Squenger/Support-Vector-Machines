"""Module returning a prediction function computed using linear regression."""

# Standard library imports
from typing import Callable

# Third party libraries imports
import numpy as np
from sklearn.linear_model import LinearRegression


def get_linear_regression(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple[Callable, dict]:
    """Returns a prediction function based on linear regression.

    Parameters
    ----------
    x_data: np.ndarray
        Input feature data.
    y_data: np.ndarray
        Target output data.

    Returns
    -------
        function: A function that takes an input array and returns predicted outputs.
    """
    # Reshape x_data if it's one-dimensional
    if x_data.ndim == 1:
        x_data = x_data.reshape(-1, 1)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x_data, y_data)

    # Define the prediction function
    def prediction_function(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return model.predict(x)

    print(
        f"Linear regression model coefficients: slope={model.coef_}, intercept={model.intercept_}"
    )

    model_dict = {
        "slope": model.coef_,
        "intercept": model.intercept_,
    }

    return prediction_function, model_dict
