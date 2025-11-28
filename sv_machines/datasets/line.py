"""Generates a dataset of points in a line shape for regression task."""

# Third party imports
import numpy as np


def get_line_dataset(slope: float, offset: float, epsilon: float, x_range: tuple[float,float] = (0,10), num_points: int = 1000, epsilon_strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Generates a dataset of points in a circle shape.

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
    
    x_vals = np.random.uniform(x_min, x_max, size=num_points)
    y_vals = slope * x_vals + offset
    if epsilon_strict:
        y_vals += np.random.uniform(-epsilon, epsilon, size=y_vals.shape)
    else:
        y_vals += np.random.normal(0, epsilon, size=y_vals.shape)

    return x_vals, y_vals