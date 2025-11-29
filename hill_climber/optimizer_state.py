"""Helper functions and dataclass for managing hill climber optimization state."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import pandas as pd
import time


@dataclass
class ReplicaState:
    """State container for a single replica in the hill climber optimization.
    
    This dataclass provides type safety and IDE autocomplete for replica state,
    replacing the previous dictionary-based approach.
    
    Attributes:
        replica_id: Replica identifier
        temperature: Current temperature
        current_data: Current data configuration
        current_objective: Current objective value
        best_data: Best data found so far
        best_objective: Best objective value found
        step: Number of accepted steps
        total_iterations: Total number of perturbations attempted
        metrics_history: List of metric dictionaries for each accepted step
        temperature_history: List of (step, temperature) tuples for temperature changes
        exchange_attempts: Total number of exchange attempts
        exchange_acceptances: Number of successful exchanges
        partner_history: List of partner replica IDs for successful exchanges
        original_data: Original input data before optimization
        hyperparameters: Optimization hyperparameters dictionary
        start_time: Unix timestamp when replica started
    """
    replica_id: int
    temperature: float
    current_data: np.ndarray
    current_objective: float
    best_data: np.ndarray
    best_objective: float
    step: int = 0
    total_iterations: int = 0
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    temperature_history: List[Tuple[int, float]] = field(default_factory=list)
    exchange_attempts: int = 0
    exchange_acceptances: int = 0
    partner_history: List[int] = field(default_factory=list)
    original_data: Optional[np.ndarray] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert ReplicaState to dictionary for backwards compatibility.
        
        Returns:
            Dictionary representation of replica state
        """
        return {
            'replica_id': self.replica_id,
            'temperature': self.temperature,
            'current_data': self.current_data,
            'current_objective': self.current_objective,
            'best_data': self.best_data,
            'best_objective': self.best_objective,
            'step': self.step,
            'total_iterations': self.total_iterations,
            'metrics_history': self.metrics_history,
            'temperature_history': self.temperature_history,
            'exchange_attempts': self.exchange_attempts,
            'exchange_acceptances': self.exchange_acceptances,
            'partner_history': self.partner_history,
            'original_data': self.original_data,
            'hyperparameters': self.hyperparameters,
            'start_time': self.start_time
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict) -> 'ReplicaState':
        """Create ReplicaState from dictionary for backwards compatibility.
        
        Args:
            state_dict: Dictionary containing replica state
            
        Returns:
            ReplicaState instance
        """
        return cls(
            replica_id=state_dict['replica_id'],
            temperature=state_dict['temperature'],
            current_data=state_dict['current_data'],
            current_objective=state_dict['current_objective'],
            best_data=state_dict['best_data'],
            best_objective=state_dict['best_objective'],
            step=state_dict.get('step', 0),
            total_iterations=state_dict.get('total_iterations', 0),
            metrics_history=state_dict.get('metrics_history', []),
            temperature_history=state_dict.get('temperature_history', []),
            exchange_attempts=state_dict.get('exchange_attempts', 0),
            exchange_acceptances=state_dict.get('exchange_acceptances', 0),
            partner_history=state_dict.get('partner_history', []),
            original_data=state_dict.get('original_data'),
            hyperparameters=state_dict.get('hyperparameters', {}),
            start_time=state_dict.get('start_time', time.time())
        )


def create_replica_state(
    replica_id: int,
    temperature: float,
    current_data: np.ndarray,
    current_objective: float,
    best_data: np.ndarray,
    best_objective: float,
    original_data: Optional[np.ndarray] = None,
    hyperparameters: Optional[Dict] = None
) -> Dict:
    """Create a new replica state dictionary (legacy function for backwards compatibility).
    
    Note: For new code, prefer using ReplicaState dataclass directly.
    
    Args:
        replica_id: Replica identifier
        temperature: Initial temperature
        current_data: Current data configuration
        current_objective: Current objective value
        best_data: Best data found so far
        best_objective: Best objective value found
        original_data: Original input data
        hyperparameters: Optimization hyperparameters
        
    Returns:
        Dictionary containing replica state
    """
    state = ReplicaState(
        replica_id=replica_id,
        temperature=temperature,
        current_data=current_data,
        current_objective=current_objective,
        best_data=best_data,
        best_objective=best_objective,
        original_data=original_data,
        hyperparameters=hyperparameters or {}
    )
    return state.to_dict()


def record_temperature_change(state: Dict, new_temperature: float, step: Optional[int] = None):
    """Record a temperature change from replica exchange.
    
    Args:
        state: Replica state dictionary
        new_temperature: New temperature after exchange
        step: Step number when exchange occurred (uses state['step'] if not provided)
    """
    exchange_step = step if step is not None else state['step']
    state['temperature_history'].append((exchange_step, new_temperature))
    state['temperature'] = new_temperature


def record_exchange(state: Dict, partner_id: int, accepted: bool):
    """Record an exchange attempt.
    
    Args:
        state: Replica state dictionary
        partner_id: ID of the partner replica
        accepted: Whether the exchange was accepted
    """
    state['exchange_attempts'] += 1
    if accepted:
        state['exchange_acceptances'] += 1
        state['partner_history'].append(partner_id)


def get_history_dataframe(state: Dict) -> pd.DataFrame:
    """Convert replica history to DataFrame.
    
    Args:
        state: Replica state dictionary
        
    Returns:
        DataFrame with step and metric columns
    """
    if not state['metrics_history']:
        return pd.DataFrame()
    
    # metrics_history is now a list of metric dicts
    # Each dict contains all metrics including 'Objective value'
    df = pd.DataFrame(state['metrics_history'])
    
    # Add step numbers (1-indexed, based on number of accepted steps)
    df.insert(0, 'Step', range(1, len(df) + 1))
    
    return df
