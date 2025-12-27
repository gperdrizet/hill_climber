"""Worker process for parallel replica optimization."""

import time
import numpy as np
from typing import Dict, Any, Tuple, Callable

from .climber_functions import perturb_vectors, calculate_objective


def run_replica_steps(
    state_dict: Dict[str, Any],
    objective_func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_steps: int,
    mode: str,
    target_value: float = None,
    db_config: Dict[str, Any] = None,
    start_time: float = None
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
        start_time (float, optional): Start time of optimization run for calculating
            time-based step spread cooling. Default is None.
    
    Returns:
        Dict[str, Any]: Updated state dictionary with new current/best states and history.
    """

    # State is already a dict
    state = state_dict
    
    # Pre-extract frequently accessed variables to avoid repeated dict lookups
    perturb_fraction = state['hyperparameters']['perturb_fraction']
    step_spread_initial = state['hyperparameters']['step_spread_absolute_initial']
    step_spread_final = state['hyperparameters'].get('step_spread_absolute_final', None)
    max_time = state['hyperparameters']['max_time']
    cooling_rate = state['hyperparameters']['cooling_rate']
    replica_id = state['replica_id']
    
    # Calculate time-based step spread cooling (applies to all features proportionally)
    if start_time is not None and step_spread_final is not None:
        elapsed_time = time.time() - start_time
        progress = min(elapsed_time / (max_time * 60.0), 1.0)  # max_time is in minutes
        # Linear interpolation from initial to final
        step_spread = step_spread_initial + (step_spread_final - step_spread_initial) * progress
    else:
        step_spread = step_spread_initial
    
    # Pre-compute mode integer for faster comparison (avoid string comparisons)
    MODE_MAXIMIZE = 0
    MODE_MINIMIZE = 1
    MODE_TARGET = 2
    mode_int = {'maximize': MODE_MAXIMIZE, 'minimize': MODE_MINIMIZE, 'target': MODE_TARGET}[mode]
    
    # Initialize database buffers if enabled
    db_enabled = db_config and db_config.get('enabled', False)
    perturbations_buffer = []
    accepted_buffer = []
    step_metrics_buffer = []
    improvements_buffer = []
    improvement_metrics_buffer = []
    
    if db_enabled:
        db_step_interval = db_config['step_interval']
    
    # Run n steps
    for iteration in range(n_steps):
        
        # Get current perturbation number and increment
        perturbation_num = state['perturbation_num']
        state['perturbation_num'] += 1
        
        # Perturb data (using pre-extracted variables)
        perturbed = perturb_vectors(
            state['current_data'],
            perturb_fraction,
            bounds,
            step_spread
        )
        
        # Evaluate
        metrics, objective = calculate_objective(
            perturbed, objective_func
        )
        
        # Acceptance criterion (simulated annealing)
        accept = _should_accept(
            objective, state['current_objective'], state['temperature'],
            mode, target_value
        )
        
        # Check if this is an improvement
        is_better = False
        if mode_int == MODE_MAXIMIZE:
            is_better = objective > state['best_objective']
        elif mode_int == MODE_MINIMIZE:
            is_better = objective < state['best_objective']
        else:  # target mode
            current_dist = abs(state['best_objective'] - target_value)
            new_dist = abs(objective - target_value)
            is_better = new_dist < current_dist
        
        # Record ALL perturbations at db_step_interval (sampled view)
        if db_enabled and (perturbation_num % db_step_interval == 0):
            timestamp = time.time()
            perturbations_buffer.append((
                replica_id, perturbation_num, objective,
                accept, is_better, state['temperature'], timestamp
            ))
        
        # If accepted, update current state and record
        if accept:
            # Update current state
            state['current_data'] = perturbed
            state['current_objective'] = objective
            state['num_accepted'] += 1
            
            # Record accepted step with full metrics
            if db_enabled:
                timestamp = time.time()
                accepted_buffer.append((
                    replica_id, perturbation_num, objective,
                    state['temperature'], timestamp
                ))
                
                # Record all user-defined metrics for this accepted step
                for metric_name, metric_value in metrics.items():
                    step_metrics_buffer.append((
                        replica_id, perturbation_num, metric_name, metric_value
                    ))
        
        # Cool temperature after every step (using pre-extracted cooling_rate)
        state['temperature'] *= (1 - cooling_rate)
        
        # If improvement, update best state and record
        if is_better:
            state['best_data'] = perturbed.copy()
            state['best_objective'] = objective
            state['best_metrics'] = metrics.copy()
            state['num_improvements'] += 1
            
            # Record improvement
            if db_enabled:
                timestamp = time.time()
                improvements_buffer.append((
                    replica_id, perturbation_num, objective,
                    state['temperature'], timestamp
                ))
                
                # Record all metrics at improvement point
                for metric_name, metric_value in metrics.items():
                    improvement_metrics_buffer.append((
                        replica_id, perturbation_num, metric_name, metric_value
                    ))
    
    # Return all buffers to main process for centralized writing
    if db_enabled:
        state['db_buffers'] = {
            'perturbations': perturbations_buffer,
            'accepted': accepted_buffer,
            'step_metrics': step_metrics_buffer,
            'improvements': improvements_buffer,
            'improvement_metrics': improvement_metrics_buffer
        }
    
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
