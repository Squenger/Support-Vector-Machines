"""Generates a dataset of points in a circle shape for regression task."""

# Third party imports
import numpy as np


def get_circle_dataset(radius: float, epsilon: float, num_points: int = 1000, epsilon_strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        radiuses = radius +np.random.normal(0, epsilon, size=num_points)

    x_vals = radiuses * np.cos(angles)
    y_vals = radiuses * np.sin(angles)
    
    return x_vals, y_vals