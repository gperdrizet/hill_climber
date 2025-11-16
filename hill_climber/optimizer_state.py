"""Dataclass for managing hill climber optimization state."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class OptimizerState:
    """Unified state management for hill climbing optimization.
    
    This dataclass consolidates all optimization tracking data into a single,
    coherent structure. It replaces multiple class attributes with a unified
    state object that simplifies data handling for plotting, checkpointing,
    and results reporting.
    """
    
    # Current state
    current_data: Optional[np.ndarray] = None
    best_data: Optional[np.ndarray] = None
    current_objective: Optional[float] = None
    best_objective: Optional[float] = None
    
    # Progress tracking
    step: int = 0
    temperature: float = 0.0
    best_distance: Optional[float] = None  # For target mode
    
    # Metrics and history
    metrics: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List] = field(default_factory=lambda: {
        'Step': [],
        'Objective value': [],
        'Best_data': []
    })
    
    # Timing
    start_time: Optional[float] = None
    last_save_time: Optional[float] = None
    last_plot_time: Optional[float] = None
    
    # Configuration
    original_data: Optional[Any] = None  # Can be numpy array or DataFrame
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


    def initialize(self, data: np.ndarray, objective: float, metrics: Dict[str, Any],
                   temperature: float, target_value: Optional[float] = None,
                   mode: str = 'maximize', original_data: Optional[Any] = None,
                   hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize state with starting data and metrics.
        
        Args:
            data: Initial data (numpy array)
            objective: Initial objective value
            metrics: Dictionary of initial metric values
            temperature: Initial temperature
            target_value: Target value for target mode (optional)
            mode: Optimization mode ('maximize', 'minimize', or 'target')
            original_data: Original input data before optimization (optional)
            hyperparameters: Dictionary of optimization hyperparameters (optional)
        """

        self.current_data = data.copy()
        self.best_data = data.copy()
        self.current_objective = objective
        self.best_objective = objective
        self.temperature = temperature
        self.metrics = metrics.copy()
        
        # Store configuration
        if original_data is not None:
            self.original_data = original_data

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters.copy()
        
        # Initialize history with metric columns
        for metric_name in metrics.keys():
            if metric_name not in self.history:
                self.history[metric_name] = []
        
        # Set best_distance for target mode
        if mode == 'target' and target_value is not None:
            self.best_distance = abs(objective - target_value)

        else:
            self.best_distance = None


    def record_improvement(self) -> None:
        """Record current best solution in history."""

        self.history['Step'].append(self.step)
        self.history['Objective value'].append(self.best_objective)
        self.history['Best_data'].append(self.best_data.copy())
        
        for metric_name, metric_value in self.metrics.items():
            self.history[metric_name].append(metric_value)


    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert history to pandas DataFrame.
        
        Returns:
            DataFrame with all history columns
        """

        return pd.DataFrame(self.history)


    def update_current(self, data: np.ndarray, objective: float, 
                      metrics: Dict[str, Any]) -> None:
        """Update current solution state.
        
        Args:
            data: New current data
            objective: New current objective value
            metrics: New current metrics
        """

        self.current_data = data
        self.current_objective = objective
        self.metrics = metrics


    def update_best(self, data: np.ndarray, objective: float,
                   target_value: Optional[float] = None) -> None:
        """Update best solution state.
        
        Args:
            data: New best data
            objective: New best objective value
            target_value: Target value for distance calculation (optional)
        """

        self.best_data = data.copy()
        self.best_objective = objective
        
        if target_value is not None:
            self.best_distance = abs(objective - target_value)


    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for checkpointing.
        
        Returns:
            Dictionary containing all state data including hyperparameters
        """

        return {
            'current_data': self.current_data.copy() if self.current_data is not None else None,
            'best_data': self.best_data.copy() if self.best_data is not None else None,
            'current_objective': self.current_objective,
            'best_objective': self.best_objective,
            'step': self.step,
            'temperature': self.temperature,
            'best_distance': self.best_distance,
            'metrics': self.metrics.copy(),
            'history': {k: v.copy() if isinstance(v, list) else v 
                       for k, v in self.history.items()},
            'start_time': self.start_time,
            'last_save_time': self.last_save_time,
            'last_plot_time': self.last_plot_time,
            'original_data': self.original_data,
            'hyperparameters': self.hyperparameters.copy() if self.hyperparameters else {}
        }


    @classmethod
    def from_checkpoint_dict(cls, checkpoint_dict: Dict[str, Any]) -> 'OptimizerState':
        """Create OptimizerState from checkpoint dictionary.
        
        Args:
            checkpoint_dict: Dictionary containing state data
            
        Returns:
            New OptimizerState instance
        """

        state = cls()
        state.current_data = checkpoint_dict.get('current_data')
        state.best_data = checkpoint_dict.get('best_data')
        state.current_objective = checkpoint_dict.get('current_objective')
        state.best_objective = checkpoint_dict.get('best_objective')
        state.step = checkpoint_dict.get('step', 0)
        state.temperature = checkpoint_dict.get('temperature', 0.0)
        state.best_distance = checkpoint_dict.get('best_distance')
        state.metrics = checkpoint_dict.get('metrics', {})
        state.history = checkpoint_dict.get('history', {
            'Step': [],
            'Objective value': [],
            'Best_data': []
        })
        state.start_time = checkpoint_dict.get('start_time')
        state.last_save_time = checkpoint_dict.get('last_save_time')
        state.last_plot_time = checkpoint_dict.get('last_plot_time')
        state.original_data = checkpoint_dict.get('original_data')
        state.hyperparameters = checkpoint_dict.get('hyperparameters', {})
        
        return state


    def has_steps(self) -> bool:
        """Check if any steps have been recorded in history.
        
        Returns:
            True if history contains steps, False otherwise
        """

        return len(self.history.get('Step', [])) > 0


    def get_results(self, is_dataframe: bool = False, 
                   columns: Optional[List[str]] = None) -> tuple:
        """Get optimization results in the appropriate format.
        
        Args:
            is_dataframe: Whether to return data as DataFrame
            columns: Column names for DataFrame (if is_dataframe=True)
            
        Returns:
            Tuple of (best_data, history_df) where:
                - best_data: Best solution (DataFrame or numpy array)
                - history_df: DataFrame with optimization history
        """

        best_data_output = (
            pd.DataFrame(self.best_data, columns=columns)
            if is_dataframe and columns is not None
            else self.best_data
        )
        
        return best_data_output, self.get_history_dataframe()
