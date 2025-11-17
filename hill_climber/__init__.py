"""Hill Climber - Optimization package with simulated annealing.

This package provides hill climbing optimization with simulated annealing,
supporting flexible multi-objective optimization and parallel execution.

Main Components:
    HillClimber: Main optimization class
    OptimizerState: State management dataclass (typically not used directly)
    Helper functions: Data manipulation and objective calculation utilities
    Plotting functions: Visualization tools for input data and results

Example:
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
    >>> best_data, steps_df = climber.climb()
    >>> 
    >>> # Or run parallel replicates
    >>> results = climber.climb_parallel(replicates=4)
    >>> 
    >>> # Visualize results
    >>> climber.plot_results(results, plot_type='histogram')
"""

__version__ = '0.1.11'
__author__ = 'gperdrizet'

from .optimizer import HillClimber
from .optimizer_state import OptimizerState
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
    'OptimizerState',
    'perturb_vectors',
    'extract_columns',
    'calculate_correlation_objective',
    'plot_input_data',
    'plot_results',
]
