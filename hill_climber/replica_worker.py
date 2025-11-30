"""Worker process for parallel replica optimization."""

import numpy as np
from typing import Dict, Any, Tuple, Callable

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
    
    This function is designed to run in a worker process. It takes
    a replica state dict, performs n optimization steps, and returns
    the updated state dict.
    
    Args:
        state_dict (Dict[str, Any]): Replica state dictionary containing current state,
            best state, hyperparameters, and history.
        objective_func (Callable): Function taking M column arrays, returns 
            (metrics_dict, objective_value).
        bounds (Tuple[np.ndarray, np.ndarray]): Tuple of (min_values, max_values) for 
            boundary reflection.
        n_steps (int): Number of optimization steps to perform.
        mode (str): Optimization mode - 'maximize', 'minimize', or 'target'.
        target_value (float, optional): Target value, only used if mode='target'. 
            Default is None.
        db_config (Dict[str, Any], optional): Optional database configuration dict with keys:
            - enabled (bool): Whether database logging is enabled.
            - path (str): Path to database file.
            - step_interval (int): Collect every Nth step.
            Default is None.
    
    Returns:
        Dict[str, Any]: Updated state dictionary with new current/best states and history.
    """

    # State is already a dict
    state = state_dict
    
    # Pre-extract frequently accessed variables to avoid repeated dict lookups
    perturb_fraction = state['hyperparameters']['perturb_fraction']
    step_spread = state['hyperparameters'].get('step_spread', 1.0)
    cooling_rate = state['hyperparameters']['cooling_rate']
    replica_id = state['replica_id']
    
    # Pre-compute mode integer for faster comparison (avoid string comparisons)
    MODE_MAXIMIZE = 0
    MODE_MINIMIZE = 1
    MODE_TARGET = 2
    mode_int = {'maximize': MODE_MAXIMIZE, 'minimize': MODE_MINIMIZE, 'target': MODE_TARGET}[mode]
    
    # Initialize database buffer if enabled (worker only collects, doesn't write)
    db_buffer = []
    db_enabled = db_config and db_config.get('enabled', False)
    
    if db_enabled:
        db_step_interval = db_config['step_interval']
    
    # Run n steps
    for iteration in range(n_steps):
        
        # Increment total iterations counter (counts all perturbations, not just accepted)
        state['total_iterations'] += 1
        
        # Perturb data (using pre-extracted variables)
        perturbed = perturb_vectors(
            state['current_data'],
            perturb_fraction,
            bounds,
            step_spread
        )
        
        # Evaluate
        metrics, objective = evaluate_objective(
            perturbed, objective_func
        )
        
        # Track metrics for potential DB write (always use most recent evaluation)
        last_metrics = metrics
        last_objective = objective
        
        # Acceptance criterion (simulated annealing)
        accept = _should_accept(
            objective, state['current_objective'], state['temperature'],
            mode, target_value
        )
        
        # Note: We only record accepted steps to avoid misleading history
        # where rejected steps would show the old objective with a new step number
        if accept:

            # Update current state
            state['current_data'] = perturbed
            state['current_objective'] = objective
            
            # Update best state if this is better
            # Use mode-aware comparison (with integer mode for speed)
            is_better = False
            if mode_int == MODE_MAXIMIZE:
                is_better = objective > state['best_objective']
            elif mode_int == MODE_MINIMIZE:
                is_better = objective < state['best_objective']
            else:  # target mode
                current_dist = abs(state['best_objective'] - target_value)
                new_dist = abs(objective - target_value)
                is_better = new_dist < current_dist
            
            if is_better:
                state['best_data'] = perturbed.copy()
                state['best_objective'] = objective
            
            state['step'] += 1
            
            # CRITICAL FIX: Record BEST objective in history, not current
            # This ensures history shows monotonic improvement
            # Add current objective as separate metric for SA analysis
            metrics['Objective value'] = state['best_objective']  # Best so far
            metrics['Current Objective'] = objective  # This step's value (may be worse due to SA)
            state['metrics_history'].append(metrics.copy())
        
            # Cool temperature (using pre-extracted cooling_rate)
            state['temperature'] *= (1 - cooling_rate)
        
        # Collect metrics for database at regular intervals (regardless of acceptance)
        # Use metrics from most recent evaluation (avoids redundant objective calls)
        if db_enabled and (iteration > 0) and (iteration % db_step_interval == 0):
            # Create a combined metrics dictionary with proper prefixes
            all_metrics = {}
            
            # Add BEST metrics with "Best " prefix (from state)
            all_metrics['Best Objective'] = state['best_objective']
            # For best metrics, use last_metrics if this was accepted, otherwise they haven't changed
            for metric_name, metric_value in last_metrics.items():
                all_metrics[f'Best {metric_name}'] = metric_value
            
            # Add CURRENT metrics with "Current " prefix (from most recent evaluation)
            all_metrics['Current Objective'] = state['current_objective']
            for metric_name, metric_value in last_metrics.items():
                all_metrics[f'Current {metric_name}'] = metric_value
            
            # Buffer all metrics for this step (using pre-extracted replica_id)
            current_step = state['step']
            for metric_name, metric_value in all_metrics.items():
                db_buffer.append((replica_id, current_step, metric_name, metric_value))
    
    # Return DB buffer to main process for centralized writing
    if db_enabled and db_buffer:
        state['db_buffer'] = db_buffer
    
    # Return state (already in dict format)
    return state


def _should_accept(
    new_obj: float,
    current_obj: float,
    temp: float,
    mode: str,
    target_value: float = None
) -> bool:
    """Determine if new state should be accepted using simulated annealing.
    
    Uses the Metropolis criterion: always accept improvements, accept
    worse solutions with probability exp(delta/T).
    
    Args:
        new_obj (float): New objective value.
        current_obj (float): Current objective value.
        temp (float): Current temperature.
        mode (str): Optimization mode - 'maximize', 'minimize', or 'target'.
        target_value (float, optional): Target value, only used if mode='target'.
            Default is None.
        
    Returns:
        bool: True if new state should be accepted, False otherwise.
    """

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
