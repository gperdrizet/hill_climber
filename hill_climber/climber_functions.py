"""Helper functions for hill climbing optimization."""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _perturb_core(data_array, step_size, n_perturb):
    """JIT-compiled core perturbation logic.
    
    This function is optimized with Numba for fast execution.
    
    Args:
        data_array: 2D numpy array to perturb
        step_size: Standard deviation of perturbation
        n_perturb: Number of elements to perturb
        
    Returns:
        Perturbed numpy array
    """
    n_rows, n_cols = data_array.shape
    result = data_array.copy()
    
    for _ in range(n_perturb):
        row_idx = np.random.randint(0, n_rows)
        col_idx = np.random.randint(0, n_cols)
        perturbation = np.random.normal(0, step_size)
        new_value = result[row_idx, col_idx] + perturbation
        
        # Ensure positive values
        result[row_idx, col_idx] = abs(new_value)
    
    return result


def perturb_vectors(data, step_size, perturb_fraction=0.1):
    """Randomly perturb a fraction of elements in the data.
    
    This function uses JIT-compiled core logic for performance.
    Works directly with numpy arrays - no DataFrame conversions.
    
    Args:
        data: Input data as numpy array
        step_size: Standard deviation of normal distribution for perturbations
        perturb_fraction: Fraction of total elements to perturb (default: 0.1)
        
    Returns:
        Perturbed numpy array
    """
    # Calculate number of elements to perturb
    n_total = data.size
    n_perturb = max(1, int(n_total * perturb_fraction))
    
    # Call JIT-compiled function
    return _perturb_core(data, step_size, n_perturb)


def extract_columns(data):
    """Extract x and y columns from numpy array.
    
    Args:
        data: numpy array (Nx2)
        
    Returns:
        Tuple of (x, y) as 1D numpy arrays
    """
    return data[:, 0], data[:, 1]


def calculate_correlation_objective(data, objective_func):
    """Calculate objective value using provided objective function.
    
    Args:
        data: Input data as numpy array (Nx2)
        objective_func: Function that takes (x, y) and returns (metrics_dict, objective_value)
        
    Returns:
        Tuple of (metrics_dict, objective_value)
    """
    x, y = extract_columns(data)
    return objective_func(x, y)