"""Helper functions for hill climbing optimization."""

import numpy as np
import pandas as pd


def perturb_vectors(data, step_size):
    '''Randomly perturb a single value in the input data.
    
    Ensures all values remain strictly positive (> 0) after perturbation.
    
    Args:
        data: numpy array or pandas DataFrame to perturb
        step_size: maximum perturbation amount
        
    Returns:
        Perturbed data in the same format as input (array or DataFrame)
    '''
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        columns = data.columns
        new_data = data.values.copy()
    else:
        new_data = data.copy()
    
    # Select and perturb random element
    flat_array = new_data.flatten()
    element_idx = np.random.randint(0, len(flat_array))
    original_value = flat_array[element_idx]
    perturbation = np.random.uniform(-step_size, step_size)
    new_value = original_value + perturbation
    
    # Ensure value stays positive
    if new_value <= 0:
        new_value = original_value + np.random.uniform(0, step_size)
    
    flat_array[element_idx] = new_value
    new_data = flat_array.reshape(new_data.shape)
    
    return pd.DataFrame(new_data, columns=columns) if is_dataframe else new_data


def extract_columns(data):
    '''Extract x and y columns from data.
    
    Args:
        data: numpy array (Nx2) or pandas DataFrame with 2 columns
        
    Returns:
        Tuple of (x, y) as arrays or Series
    '''
    if isinstance(data, pd.DataFrame):
        cols = data.columns.tolist()
        return data[cols[0]], data[cols[1]]
    else:
        return data[:, 0], data[:, 1]


def calculate_correlation_objective(data, objective_func):
    '''Calculate objective function using the specified objective.
    
    Args:
        data: numpy array (Nx2) or pandas DataFrame with 2 columns
        objective_func: function that calculates the objective
        
    Returns:
        Tuple of (metrics_dict, objective_value)
    '''
    x, y = extract_columns(data)
    return objective_func(x, y)