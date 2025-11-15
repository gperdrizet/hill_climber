"""Helper functions for hill climbing optimization."""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _perturb_core(data_array, step_size, n_perturb, min_bounds, max_bounds, perturb_row):
    """JIT-compiled core perturbation logic.
    
    Uses reflection to keep values within bounds instead of clipping,
    which prevents pile-up of values at the boundaries.
    
    Args:
        data_array: 2D numpy array to perturb
        step_size: Half-width of uniform distribution for perturbation
        n_perturb: Number of elements (or rows) to perturb
        min_bounds: 1D array of minimum bounds for each column
        max_bounds: 1D array of maximum bounds for each column
        perturb_row: If True, perturb entire rows; if False, perturb individual elements
        
    Returns:
        Perturbed numpy array
    """

    n_rows, n_cols = data_array.shape
    result = data_array.copy()
    
    if perturb_row:
        # Perturb entire rows
        for _ in range(n_perturb):
            row_idx = np.random.randint(0, n_rows)
            
            # Perturb all columns in this row
            for col_idx in range(n_cols):
                perturbation = np.random.uniform(-step_size, step_size)
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
    else:
        # Perturb individual elements
        for _ in range(n_perturb):
            row_idx = np.random.randint(0, n_rows)
            col_idx = np.random.randint(0, n_cols)
            perturbation = np.random.uniform(-step_size, step_size)
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


def perturb_vectors(data, step_size, perturb_fraction=0.1, bounds=None, perturb_row=False):
    """Randomly perturb a fraction of elements in the data.
    
    This function uses JIT-compiled core logic for performance.
    Works directly with numpy arrays - no DataFrame conversions.
    
    Args:
        data: Input data as numpy array
        step_size: Half-width of uniform distribution for perturbations
        perturb_fraction: Fraction of total elements (or rows) to perturb (default: 0.1)
        bounds: Tuple of (min_bounds, max_bounds) arrays for each column.
                If None, uses data min/max (default: None)
        perturb_row: If True, perturb entire rows; if False, perturb individual elements (default: False)
        
    Returns:
        Perturbed numpy array
    """

    # Calculate number of elements or rows to perturb
    if perturb_row:
        n_total = data.shape[0]  # Number of rows
    else:
        n_total = data.size  # Total number of elements
    
    n_perturb = max(1, int(n_total * perturb_fraction))
    
    # Determine bounds
    if bounds is None:
        min_bounds = np.min(data, axis=0)
        max_bounds = np.max(data, axis=0)

    else:
        min_bounds, max_bounds = bounds
    
    # Call JIT-compiled function
    return _perturb_core(data, step_size, n_perturb, min_bounds, max_bounds, perturb_row)


def extract_columns(data):
    """Extract columns from numpy array.
    
    Works with n-dimensional data by returning each column separately.
    
    Args:
        data: numpy array (N x M) where N is number of samples, M is number of features
        
    Returns:
        Tuple of 1D numpy arrays, one for each column
        
    Examples:
        For 2D data (N x 2): returns (x, y)
        For 3D data (N x 3): returns (x, y, z)
        For nD data (N x M): returns (col0, col1, ..., colM-1)
    """

    return tuple(data[:, i] for i in range(data.shape[1]))


def calculate_objective(data, objective_func):
    """Calculate objective value using provided objective function.
    
    Extracts columns from data and passes them to the objective function.
    Supports n-dimensional data.
    
    Args:
        data: Input data as numpy array (N x M)
        objective_func: Function that takes M column arrays and returns 
                       (metrics_dict, objective_value)
        
    Returns:
        Tuple of (metrics_dict, objective_value)
        
    Examples:
        For 2D data: objective_func(x, y) is called
        For 3D data: objective_func(x, y, z) is called
        For nD data: objective_func(col0, col1, ...) is called
    """

    columns = extract_columns(data)

    return objective_func(*columns)


# Backwards compatibility alias
calculate_correlation_objective = calculate_objective