"""Hill Climber - Parallel optimization with replica exchange.

This package provides hill climbing optimization using replica exchange 
(parallel tempering) for improved global optimization performance.

Main Components:
    HillClimber: Main optimization class with replica exchange
    TemperatureLadder: Temperature ladder for replica exchange
    ExchangeScheduler: Scheduler for replica exchange attempts
    ReplicaState: State container for individual replicas
    OptimizerConfig: Type-safe configuration dataclass
    Helper functions: Data manipulation and objective calculation utilities
    Plotting functions: Visualization tools for input data and results

Example:
    >>> from hill_climber import HillClimber, plot_optimization_results
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
    >>> plot_optimization_results(climber, plot_type='scatter')
    >>> # Or from a checkpoint file
    >>> plot_optimization_results('checkpoint.pkl', all_replicas=True)
"""

__version__ = '2.1.2'
__author__ = 'gperdrizet'

from .optimizer import HillClimber
from .config import OptimizerConfig
from .optimizer_state import ReplicaState, create_replica_state
from .replica_exchange import (
    TemperatureLadder,
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
    plot_optimization_results
)

__all__ = [
    'HillClimber',
    'OptimizerConfig',
    'ReplicaState',
    'create_replica_state',
    'TemperatureLadder',
    'ExchangeScheduler',
    'perturb_vectors',
    'extract_columns',
    'calculate_objective',
    'evaluate_objective',
    'plot_input_data',
    'plot_optimization_results',
]
