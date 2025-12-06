import numpy as np

def epsilon_insensitive(xi, epsilon=1.0):
    """
    Epsilon-insensitive loss:
    0 if |xi| <= epsilon
    |xi| - epsilon otherwise
    """
    return np.maximum(0, np.abs(xi) - epsilon)

def laplacian(xi):
    """
    Laplacian loss: |xi|
    """
    return np.abs(xi)

def gaussian(xi):
    """
    Gaussian loss: 0.5 * xi^2
    """
    return 0.5 * xi**2

def huber_robust(xi, sigma=1.0):
    """
    Huber's robust loss:
    (1 / 2*sigma) * xi^2       if |xi| <= sigma
    |xi| - sigma / 2           otherwise
    """
    abs_xi = np.abs(xi)
    mask_le_sigma = abs_xi <= sigma
    
    # Calculation for |xi| <= sigma
    loss_le = (1.0 / (2.0 * sigma)) * (xi**2)
    
    # Calculation for |xi| > sigma
    loss_gt = abs_xi - (sigma / 2.0)
    
    return np.where(mask_le_sigma, loss_le, loss_gt)

def polynomial(xi, p=3.0):
    """
    Polynomial loss: (1/p) * |xi|^p
    """
    return (1.0 / p) * np.power(np.abs(xi), p)

def piecewise_polynomial(xi, sigma=1.0, p=3.0):
    """
    Piecewise polynomial loss:
    (1 / (p * sigma^(p-1))) * |xi|^p     if |xi| <= sigma
    |xi| - sigma * (p-1)/p               otherwise
    """
    abs_xi = np.abs(xi)
    mask_le_sigma = abs_xi <= sigma
    
    # Calculation for |xi| <= sigma
    factor = 1.0 / (p * np.power(sigma, p - 1))
    loss_le = factor * np.power(abs_xi, p)
    
    # Calculation for |xi| > sigma
    constant = sigma * (p - 1.0) / p
    loss_gt = abs_xi - constant
    
    return np.where(mask_le_sigma, loss_le, loss_gt)