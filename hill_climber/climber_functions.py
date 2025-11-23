"""Helper functions for hill climbing optimization."""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _perturb_core(data_array, step_spread, n_perturb, min_bounds, max_bounds):
    """JIT-compiled core perturbation logic.
    
    Uses reflection to keep values within bounds instead of clipping,
    which prevents pile-up of values at the boundaries.
    
    Args:
        data_array: 2D numpy array to perturb
        step_spread: Standard deviation of normal distribution for perturbation
        n_perturb: Number of elements to perturb
        min_bounds: 1D array of minimum bounds for each column
        max_bounds: 1D array of maximum bounds for each column
        
    Returns:
        Perturbed numpy array
    """

    n_rows, n_cols = data_array.shape
    result = data_array.copy()
    
    # Perturb individual elements
    for _ in range(n_perturb):

        row_idx = np.random.randint(0, n_rows)
        col_idx = np.random.randint(0, n_cols)
        perturbation = np.random.normal(0.0, step_spread)
        new_value = result[row_idx, col_idx] + perturbation
        
        # Reflect values back into bounds instead of clipping
        min_val = min_bounds[col_idx]
        max_val = max_bounds[col_idx]
        range_val = max_val - min_val
        
        # Reflect below minimum
        if new_value < min_val:
            new_value = min_val + (min_val - new_value)
        
        # Reflect above maximum
        if new_value > max_val:
            new_value = max_val - (new_value - max_val)
        
        # Handle cases where reflection itself goes out of bounds (large perturbations)
        # Use modulo wrapping as fallback
        if new_value < min_val or new_value > max_val:
            new_value = min_val + np.fmod(new_value - min_val, range_val)

            if new_value < min_val:
                new_value += range_val
        
        result[row_idx, col_idx] = new_value
    
    return result


def perturb_vectors(data, perturb_fraction=0.1, bounds=None, step_spread=1.0):
    """Randomly perturb a fraction of elements in the data.
    
    This function uses JIT-compiled core logic for performance.
    Works directly with numpy arrays - no DataFrame conversions.
    Perturbations are sampled from a normal distribution with mean 0.
    
    Args:
        data: Input data as numpy array
        perturb_fraction: Fraction of total elements to perturb (default: 0.1).
                         Note: HillClimber uses 0.05 as its default.
        bounds: Tuple of (min_bounds, max_bounds) arrays for each column.
                If None, uses data min/max (default: None)
        step_spread: Standard deviation of normal distribution for perturbations (default: 1.0)
        
    Returns:
        Perturbed numpy array
    """

    # Calculate number of elements to perturb
    n_total = data.size  # Total number of elements
    n_perturb = max(1, int(n_total * perturb_fraction))
    
    # Determine bounds
    if bounds is None:
        min_bounds = np.min(data, axis=0)
        max_bounds = np.max(data, axis=0)

    else:
        min_bounds, max_bounds = bounds
    
    # Call JIT-compiled function
    return _perturb_core(data, step_spread, n_perturb, min_bounds, max_bounds)


def extract_columns(data):
    """Extract columns from numpy array.
    
    Works with multi-column data by returning each column separately.
    
    Args:
        data: numpy array with shape (N, M) where N = samples, M = features
        
    Returns:
        Tuple of 1D numpy arrays, one for each column
        
    Examples:
        For 2-column data (N, 2): returns (x, y)
        For 3-column data (N, 3): returns (x, y, z)
        For M-column data (N, M): returns (col0, col1, ..., colM-1)
    """

    return tuple(data[:, i] for i in range(data.shape[1]))


def calculate_objective(data, objective_func):
    """Calculate objective value using provided objective function.
    
    Extracts columns from data and passes them to the objective function.
    Supports multi-column data.
    
    Args:
        data: Input data as numpy array with shape (N, M)
        objective_func: Function that takes M column arrays and returns 
                       (metrics_dict, objective_value)
        
    Returns:
        Tuple of (metrics_dict, objective_value)
        
    Examples:
        For 2-column data: objective_func(x, y) is called
        For 3-column data: objective_func(x, y, z) is called
        For M-column data: objective_func(col0, col1, ...) is called
    """

    columns = extract_columns(data)

    return objective_func(*columns)


def evaluate_objective(data, objective_func):
    """Evaluate objective function on data.
    
    This is a convenience wrapper around calculate_objective.
    
    Args:
        data: numpy array (N, M)
        objective_func: Function taking M arrays, returns (metrics_dict, objective_value)
        
    Returns:
        Tuple of (metrics_dict, objective_value)
    """
    return calculate_objective(data, objective_func)
