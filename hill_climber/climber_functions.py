"""Helper functions for hill climbing optimization."""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _perturb_core(data_array, step_spread, n_perturb, min_bounds, max_bounds):
    """JIT-compiled core perturbation logic.
    
    Uses reflection to keep values within bounds instead of clipping,
    which prevents pile-up of values at the boundaries.
    
    Args:
        data_array (np.ndarray): 2D numpy array to perturb with shape (N, M).
        step_spread (float): Standard deviation of normal distribution for perturbation.
        n_perturb (int): Number of elements to perturb.
        min_bounds (np.ndarray): 1D array of minimum bounds for each column.
        max_bounds (np.ndarray): 1D array of maximum bounds for each column.
        
    Returns:
        np.ndarray: Perturbed numpy array with same shape as input.
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
        data (np.ndarray): Input data as numpy array with shape (N, M).
        perturb_fraction (float): Fraction of total elements to perturb. Default is 0.1.
            Note: HillClimber uses 0.001 as its default.
        bounds (tuple, optional): Tuple of (min_bounds, max_bounds) arrays for each column.
            If None, uses data min/max. Default is None.
        step_spread (float): Standard deviation of normal distribution for perturbations.
            Default is 1.0.
        
    Returns:
        np.ndarray: Perturbed numpy array with same shape as input.
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
        data (np.ndarray): Numpy array with shape (N, M) where N = samples, M = features.
        
    Returns:
        tuple: Tuple of 1D numpy arrays, one for each column.
        
    Examples:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> x, y = extract_columns(data)
        >>> print(x)  # [1, 3, 5]
        >>> print(y)  # [2, 4, 6]
    """

    return tuple(data[:, i] for i in range(data.shape[1]))


def calculate_objective(data, objective_func):
    """Calculate objective value using provided objective function.
    
    Extracts columns from data and passes them to the objective function.
    Supports multi-column data.
    
    Args:
        data (np.ndarray): Input data as numpy array with shape (N, M).
        objective_func (Callable): Function that takes M column arrays and returns 
            (metrics_dict, objective_value).
        
    Returns:
        tuple: Tuple of (metrics_dict, objective_value) where metrics_dict is a
            dictionary of metric names to values, and objective_value is a float.
        
    Examples:
        >>> def obj_func(x, y):
        ...     return {'mean_x': x.mean()}, x.mean() + y.mean()
        >>> data = np.array([[1, 2], [3, 4]])
        >>> metrics, objective = calculate_objective(data, obj_func)
    """

    columns = extract_columns(data)

    return objective_func(*columns)


def evaluate_objective(data, objective_func):
    """Evaluate objective function on data.
    
    This is a convenience wrapper around calculate_objective.
    
    Args:
        data (np.ndarray): Numpy array with shape (N, M).
        objective_func (Callable): Function taking M arrays, returns 
            (metrics_dict, objective_value).
        
    Returns:
        tuple: Tuple of (metrics_dict, objective_value) where metrics_dict is a
            dictionary of metric names to values, and objective_value is a float.
    """
    return calculate_objective(data, objective_func)
