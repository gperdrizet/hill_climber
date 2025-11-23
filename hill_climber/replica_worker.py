"""Worker process for parallel replica optimization."""

import numpy as np
from typing import Dict, Any, Tuple, Callable

from .optimizer_state import OptimizerState
from .climber_functions import perturb_vectors, evaluate_objective


def run_replica_steps(
    state_dict: Dict[str, Any],
    objective_func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_steps: int,
    mode: str,
    target_value: float = None,
    db_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run n optimization steps for a single replica.
    
    This function is designed to run in a worker process. It deserializes
    a replica state, performs n optimization steps, and returns the updated
    serialized state.
    
    Args:
        state_dict: Serialized OptimizerState dictionary
        objective_func: Function taking M column arrays, returns (metrics_dict, objective_value)
        bounds: Tuple of (min_values, max_values) for boundary reflection
        n_steps: Number of optimization steps to perform
        mode: 'maximize', 'minimize', or 'target'
        target_value: Target value (only used if mode='target')
        db_config: Optional database configuration dict with keys:
                  - enabled: bool
                  - path: str
                  - step_interval: int (collect every Nth step)
                  - buffer_size: int (flush after N collected steps)
    
    Returns:
        Updated serialized state dictionary with 'db_buffer' key if database enabled
    """
    # Deserialize state
    state = OptimizerState(**state_dict)
    
    # Initialize database buffer if enabled
    db_buffer = []
    db_enabled = db_config and db_config.get('enabled', False)
    
    if db_enabled:
        db_step_interval = db_config['step_interval']
        db_buffer_size = db_config['buffer_size']
        db_path = db_config['path']
        db_pool_size = db_config.get('pool_size', 4)
        
        # Import database writer only if needed
        from .database import DatabaseWriter
        db_writer = DatabaseWriter(db_path, db_pool_size)
    
    # Run n steps
    for _ in range(n_steps):
        # Get step spread from hyperparameters
        step_spread = state.hyperparameters.get('step_spread', 1.0)
        
        # Perturb data
        perturb_fraction = state.hyperparameters['perturb_fraction']
        perturbed = perturb_vectors(
            state.current_data,
            perturb_fraction,
            bounds,
            step_spread
        )
        
        # Evaluate
        metrics, objective = evaluate_objective(
            perturbed, objective_func
        )
        
        # Add objective value to metrics dict for database storage
        metrics['Objective value'] = objective
        
        # Acceptance criterion (simulated annealing)
        accept = _should_accept(
            objective, state.current_objective, state.temperature,
            mode, target_value
        )
        
        if accept:
            state.record_improvement(perturbed, objective, metrics)
            
            # Collect metrics for database if enabled and on collection interval
            if db_enabled and (state.step % db_step_interval == 0):
                # Buffer metrics for this step
                for metric_name, metric_value in metrics.items():
                    db_buffer.append((state.replica_id, state.step, metric_name, metric_value))
                
                # Flush buffer if it reaches buffer_size
                if len(db_buffer) >= db_buffer_size * len(metrics):
                    db_writer.insert_metrics_batch(db_buffer)
                    db_buffer = []
        # Note: We only record accepted steps to avoid misleading history
        # where rejected steps would show the old objective with a new step number
        
        # Cool temperature
        cooling_rate = state.hyperparameters['cooling_rate']
        state.temperature *= (1 - cooling_rate)
    
    # Serialize and return
    result = _serialize_state(state)
    
    # Include unflushed buffer for main process to handle
    if db_enabled:
        result['db_buffer'] = db_buffer
    
    return result


def _should_accept(
    new_obj: float,
    current_obj: float,
    temp: float,
    mode: str,
    target_value: float = None
) -> bool:
    """Determine if new state should be accepted (simulated annealing)."""
    if mode == 'maximize':
        delta = new_obj - current_obj
    elif mode == 'minimize':
        delta = current_obj - new_obj
    else:  # target mode
        current_dist = abs(current_obj - target_value)
        new_dist = abs(new_obj - target_value)
        delta = current_dist - new_dist
    
    if delta > 0:
        return True
    else:
        prob = np.exp(delta / temp) if temp > 0 else 0
        return np.random.random() < prob


def _serialize_state(state: OptimizerState) -> Dict[str, Any]:
    """Convert OptimizerState to dictionary for pickling."""
    return {
        'replica_id': state.replica_id,
        'temperature': state.temperature,
        'current_data': state.current_data,
        'current_objective': state.current_objective,
        'best_data': state.best_data,
        'best_objective': state.best_objective,
        'step': state.step,
        'metrics_history': state.metrics_history,
        'exchange_attempts': state.exchange_attempts,
        'exchange_acceptances': state.exchange_acceptances,
        'partner_history': state.partner_history,
        'original_data': state.original_data,
        'hyperparameters': state.hyperparameters,
        'start_time': state.start_time
    }
