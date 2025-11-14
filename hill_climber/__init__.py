"""Hill Climber - Optimization package with simulated annealing.

This package provides tools for hill climbing optimization with optional
simulated annealing, supporting multiple optimization modes and parallel
execution.

Main Components:
    HillClimber: Main optimization class
    Helper functions: Data manipulation and objective calculation utilities
    Plotting functions: Visualization tools for input data and results

Example:
    >>> import sys
    >>> sys.path.insert(0, 'path/to/hill_climber/src')
    >>> 
    >>> from hill_climber import HillClimber
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'x': np.random.rand(100),
    ...     'y': np.random.rand(100)
    ... })
    >>> 
    >>> # Define objective function
    >>> def my_objective(x, y):
    ...     correlation = pd.Series(x).corr(pd.Series(y))
    ...     return {'correlation': correlation}, correlation
    >>> 
    >>> # Create and run optimizer
    >>> climber = HillClimber(
    ...     data=data,
    ...     objective_func=my_objective,
    ...     max_time=1,
    ...     mode='maximize'
    ... )
    >>> results = climber.climb_parallel(replicates=4)
    >>> 
    >>> # Visualize input data
    >>> climber.plot_input()  # Default scatter plot
    >>> climber.plot_input(plot_type='kde')  # KDE plot
    >>> 
    >>> # Plot results with scatter plots (default)
    >>> climber.plot_results(results)
    >>> 
    >>> # Or plot results with KDE (Kernel Density Estimation) plots
    >>> climber.plot_results(results, plot_type='histogram')
    >>> 
    >>> # Or select specific metrics to display
    >>> climber.plot_results(results, metrics=['correlation'])

Module Contents:
    - optimizer: Main HillClimber class module
    - climber_functions: Helper functions for data manipulation
    - plotting_functions: Visualization utilities
"""

__version__ = '0.1.1'
__author__ = 'gperdrizet'

from .optimizer import HillClimber
from .climber_functions import (
    perturb_vectors,
    extract_columns,
    calculate_correlation_objective
)
from .plotting_functions import (
    plot_input_data,
    plot_results
)

__all__ = [
    'HillClimber',
    'perturb_vectors',
    'extract_columns',
    'calculate_correlation_objective',
    'plot_input_data',
    'plot_results',
]
