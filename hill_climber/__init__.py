"""Hill Climber - Parallel optimization with replica exchange.

This package provides hill climbing optimization using replica exchange 
(parallel tempering) for improved global optimization performance.

Main Components:
    HillClimber: Main optimization class with replica exchange
    OptimizerState: Replica state management dataclass
    TemperatureLadder: Temperature ladder for replica exchange
    ExchangeStatistics: Track exchange acceptance rates
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
    >>> # Create and run optimizer with replica exchange
    >>> climber = HillClimber(
    ...     data=data,
    ...     objective_func=my_objective,
    ...     max_time=1,
    ...     mode='maximize',
    ...     n_replicas=4
    ... )
    >>> best_data, steps_df = climber.climb()
    >>> 
    >>> # Visualize results
    >>> climber.plot_results((best_data, steps_df), plot_type='histogram')
"""

__version__ = '1.0.0'
__author__ = 'gperdrizet'

from .optimizer import HillClimber
from .optimizer_state import OptimizerState
from .replica_exchange import (
    TemperatureLadder,
    ExchangeStatistics,
    ExchangeScheduler
)
from .climber_functions import (
    perturb_vectors,
    extract_columns,
    calculate_objective,
    evaluate_objective
)
from .plotting_functions import (
    plot_input_data,
    plot_results
)

__all__ = [
    'HillClimber',
    'OptimizerState',
    'TemperatureLadder',
    'ExchangeStatistics',
    'ExchangeScheduler',
    'perturb_vectors',
    'extract_columns',
    'calculate_objective',
    'evaluate_objective',
    'plot_input_data',
    'plot_results',
]
