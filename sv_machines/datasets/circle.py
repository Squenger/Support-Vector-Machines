"""Generates a dataset of points in a circle shape for regression task."""

# Third party imports
import numpy as np
import matplotlib.pyplot as plt


def get_circle_dataset(
    radius: float, epsilon: float, num_points: int = 1000, epsilon_strict: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a dataset of points in a circle shape.

    Parameters
    ----------
    radius: float
        Radius of the circle.
    epsilon: float
        Noise level to be added to the points.
    num_points: int
        Number of points to generate.
    epsilon_strict: bool
        If True, noise is generated uniformly in the range [-epsilon, epsilon].
        Therefore the datapoints will not be outside of the +/- epsilon band
        around the line.
    epsilon_strict: bool
        If True, noise is generated uniformly in the range [-epsilon, epsilon].
        Therefore the datapoints will not be outside of the +/- epsilon band
        around the line. If False, noise is generated from a normal distribution
        with mean 0 and standard deviation epsilon around the "true circle", with
        therefore the possibilitw that some points are outside the +/- epsilon
        range of the "true circle".

    Returns
    -------
    np.ndarray
        x values of the points of the dataset.
    np.ndarray
        y values of the points of the dataset.
    """
    # angles = np.linspace(0, 2 * np.pi, num_points)
    angles = np.random.uniform(0, 2 * np.pi, size=num_points)
    if epsilon_strict:
        radiuses = radius + np.random.uniform(-epsilon, epsilon, size=num_points)
    else:
        radiuses = radius + np.random.normal(0, epsilon, size=num_points)

    x_vals = radiuses * np.cos(angles)
    y_vals = radiuses * np.sin(angles)

    return x_vals, y_vals


def get_circle_plot(
    x_dataset: np.ndarray,
    y_dataset: np.ndarray,
    radius: float,
    epsilon: float,
    with_true_function: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Generates a plot of the circle dataset.

    Parameters
    ----------
    x_dataset : np.ndarray
        X values of the dataset points.
    y_dataset : np.ndarray
        Y values of the dataset points.
    radius : float
        Radius of the "true" circle.
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

    x_angle_true = np.linspace(0, 2 * np.pi, 100)
    x_axis = radius * np.cos(x_angle_true)
    y_true = radius * np.sin(x_angle_true)
    x_upper_limit = (radius + epsilon) * np.cos(x_angle_true)
    y_upper_limit = (radius + epsilon) * np.sin(x_angle_true)
    x_lower_limit = (radius - epsilon) * np.cos(x_angle_true)
    y_lower_limit = (radius - epsilon) * np.sin(x_angle_true)

    fig, ax = plt.subplots(1, 1)
    if with_true_function:
        ax.plot(
            x_axis,
            y_true,
            color="red",
            linestyle="-",
            label="True y values (without noise)",
        )
        ax.plot(
            x_upper_limit,
            y_upper_limit,
            color="green",
            linestyle="--",
            linewidth=2,
            label="True y values + epsilon",
        )
        ax.plot(
            x_lower_limit,
            y_lower_limit,
            color="green",
            linestyle="--",
            linewidth=2,
            label="True y values - epsilon",
        )
    # ax.fill(np.concatenate([x_upper_limit, x_lower_limit[::-1]]),
    #         np.concatenate([y_upper_limit, y_lower_limit[::-1]]),
    #         color='black', alpha=0.2, label=f'±{epsilon} band')
    ax.scatter(
        x_dataset,
        y_dataset,
        marker="x",
        s=8,
        color="blue",
        label="Circle dataset points",
    )
    ax.set_aspect("equal")
    fig.tight_layout()

    return fig, ax
