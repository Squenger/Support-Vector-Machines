"""Generates non linear function dataset."""

# Third party imports
import numpy as np
import matplotlib.pyplot as plt


def function_a(x: np.ndarray) -> np.ndarray:
    """Computes the value of function A at given x values.

    Parameters
    ----------
    x : np.ndarray
        Input x values.

    Returns
    -------
    np.ndarray
        Computed y values.
    """
    return x * np.log(x) * np.cos(x)


def get_function_a_dataset(
    epsilon: float,
    x_range: tuple[float, float] = (0.5, 10),
    num_points: int = 1000,
    epsilon_strict: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a dataset for a linear function of a given set.

    Parameters
    ----------
    slope: float
        Slope of the line.
    offset: float
        Offset of the line.
    epsilon: float
        Noise level to be added to the points.
    x_range: tuple[float,float]
        Range of x values to generate points from.
    num_points: int
        Number of points to generate.
    epsilon_strict: bool
        If True, noise is generated uniformly in the range [-epsilon, epsilon].
        Therefore the datapoints will not be outside of the +/- epsilon band
        around the line. If False, noise is generated from a normal distribution
        with mean 0 and standard deviation epsilon around the "true line", with
        therefore the possibilitw that some points are outside the +/- epsilon
        range of the "true line".

    Returns
    -------
    np.ndarray
        x values of the points of the dataset.
    np.ndarray
        y values of the points of the dataset.
    """
    x_min, x_max = x_range
    if x_min >= x_max:
        raise ValueError("x_range min must be less than max.")
    if x_min <= 0:
        raise ValueError("x_range min must be greater than 0 for log function.")

    x_vals = np.random.uniform(x_min, x_max, size=num_points)
    y_vals = function_a(x_vals)
    if epsilon_strict:
        y_vals += np.random.uniform(-epsilon, epsilon, size=y_vals.shape)
    else:
        y_vals += np.random.normal(0, epsilon, size=y_vals.shape)

    return x_vals, y_vals


def get_function_a_plot(
    x_dataset: np.ndarray,
    y_dataset: np.ndarray,
    epsilon: float,
    with_true_function: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Returns a matplotlib figure and axes for plotting linear regression results.

    Parameters
    ----------
    x_dataset : np.ndarray
        X values of the dataset points.
    y_dataset : np.ndarray
        Y values of the dataset points.
    slope : float
        Slope of the "true" linear function.
    offset : float
        Offset of the "true" linear function.
    epsilon : float
        Epsilon value defining the tolerance band around the true function.
    with_true_function : bool, optional
        If True, the true linear function is also plotted, by default False.

    Returns
    -------
    matplotlib.pyplot.Figure
        Handle of the matplotlib figure.
    matplotlib.pyplot.Axes
        Handle of the matplotlib axes.
    """
    # Define
    x_ordered = np.linspace(min(x_dataset), max(x_dataset), 50)
    y_true = function_a(x_ordered)
    y_upper_limit = y_true + epsilon
    y_lower_limit = y_true - epsilon

    fig, ax = plt.subplots(1, 1)
    if with_true_function:
        ax.plot(
            x_ordered,
            y_true,
            color="red",
            linestyle="-",
            label="True y values (without noise)",
        )
        # ax.plot(x_ordered, y_upper_limit, color='green', linestyle='--', linewidth=2, label='True y values + epsilon')
        # ax.plot(x_ordered, y_lower_limit, color='green', linestyle='--', linewidth=2, label='True y values - epsilon')
    ax.fill_between(
        x_ordered,
        y_lower_limit,
        y_upper_limit,
        color="black",
        alpha=0.1,
        label="+/-epsilon band",
    )
    ax.scatter(
        x_dataset, y_dataset, marker="x", s=8, color="blue", label="Line dataset points"
    )
    fig.tight_layout()
    # fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig, ax
