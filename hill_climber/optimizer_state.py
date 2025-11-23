"""Dataclass for managing hill climber optimization state."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import time


@dataclass
class OptimizerState:
    """State for a single replica in replica exchange optimization.
    
    This dataclass manages the state and history for a single optimization replica,
    including current configuration, best found solution, metrics history, and 
    exchange statistics.
    """
    
    replica_id: int
    temperature: float
    current_data: np.ndarray
    current_objective: float
    best_data: np.ndarray
    best_objective: float
    step: int = 0
    metrics_history: List[tuple] = field(default_factory=list)
    temperature_history: List[tuple] = field(default_factory=list)  # (step, new_temperature)
    exchange_attempts: int = 0
    exchange_acceptances: int = 0
    partner_history: List[int] = field(default_factory=list)
    original_data: Optional[np.ndarray] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def record_step(self, metrics_dict: Dict, objective_value: float):
        """Record a step in the optimization.
        
        Args:
            metrics_dict: Dictionary of metric values
            objective_value: Current objective value
        """
        self.metrics_history.append((self.step, metrics_dict, objective_value, self.best_data.copy()))
        self.step += 1
    
    def record_improvement(self, new_data: np.ndarray, new_objective: float, 
                          metrics_dict: Dict):
        """Record an improvement (accepted step).
        
        Args:
            new_data: New data configuration
            new_objective: New objective value
            metrics_dict: Dictionary of metric values
        """
        self.current_data = new_data.copy()
        self.current_objective = new_objective
        
        # Update best if this is better
        if self._is_better(new_objective, self.best_objective):
            self.best_data = new_data.copy()
            self.best_objective = new_objective
        
        self.record_step(metrics_dict, new_objective)
    
    def record_exchange(self, partner_id: int, accepted: bool):
        """Record an exchange attempt.
        
        Args:
            partner_id: ID of the partner replica
            accepted: Whether the exchange was accepted
        """
        self.exchange_attempts += 1
        if accepted:
            self.exchange_acceptances += 1
            self.partner_history.append(partner_id)
    
    def record_temperature_change(self, new_temperature: float, step: Optional[int] = None):
        """Record a temperature change from replica exchange.
        
        Args:
            new_temperature: New temperature after exchange
            step: Step number when exchange occurred (uses self.step if not provided)
        """
        exchange_step = step if step is not None else self.step
        self.temperature_history.append((exchange_step, new_temperature))
        self.temperature = new_temperature
    
    def _is_better(self, new_val: float, current_val: float) -> bool:
        """Check if new value is better based on mode.
        
        Args:
            new_val: New objective value
            current_val: Current best objective value
            
        Returns:
            True if new value is better
        """
        mode = self.hyperparameters.get('mode', 'maximize')
        if mode == 'maximize':
            return new_val > current_val
        elif mode == 'minimize':
            return new_val < current_val
        else:  # target mode
            target = self.hyperparameters.get('target_value', 0)
            return abs(new_val - target) < abs(current_val - target)
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame.
        
        Returns:
            DataFrame with step, objective, metric columns, and best_data snapshots
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        steps, metrics_dicts, objectives, best_data_snapshots = zip(*self.metrics_history)
        
        df_data = {
            'Step': steps,
            'Objective value': objectives,
            'Best_data': best_data_snapshots,
        }
        
        # Add metric columns
        if metrics_dicts[0]:
            for key in metrics_dicts[0].keys():
                df_data[key] = [m[key] for m in metrics_dicts]
        
        return pd.DataFrame(df_data)
    
    def get_acceptance_rate(self) -> float:
        """Get exchange acceptance rate.
        
        Returns:
            Acceptance rate as a fraction
        """
        if self.exchange_attempts == 0:
            return 0.0
        return self.exchange_acceptances / self.exchange_attempts
