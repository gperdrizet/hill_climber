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
        state_dict: Replica state dictionary
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
        Updated state dictionary
    """

    # State is already a dict
    state = state_dict
    
    # Initialize database buffer if enabled
    db_buffer = []
    db_enabled = db_config and db_config.get('enabled', False)
    
    if db_enabled:
        db_step_interval = db_config['step_interval']
        db_buffer_size = db_config['buffer_size']
        db_path = db_config['path']
        
        # Import database writer only if needed
        from .database import DatabaseWriter
        db_writer = DatabaseWriter(db_path)
    
    # Run n steps
    for iteration in range(n_steps):
        
        # Increment total iterations counter (counts all perturbations, not just accepted)
        state['total_iterations'] += 1

        # Get step spread from hyperparameters
        step_spread = state['hyperparameters'].get('step_spread', 1.0)
        
        # Perturb data
        perturb_fraction = state['hyperparameters']['perturb_fraction']

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
            # Use mode-aware comparison
            is_better = False
            if mode == 'maximize':
                is_better = objective > state['best_objective']
            elif mode == 'minimize':
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
        
            # Cool temperature
            cooling_rate = state['hyperparameters']['cooling_rate']
            state['temperature'] *= (1 - cooling_rate)
        
        # Collect metrics for database at regular intervals (regardless of acceptance)
        # Record both BEST and CURRENT state metrics for dashboard flexibility
        if db_enabled and (iteration > 0) and (iteration % db_step_interval == 0):
            # Re-evaluate best data to get its metrics
            best_metrics, best_obj = evaluate_objective(
                state['best_data'], objective_func
            )
            
            # Re-evaluate current data to get its metrics (may differ due to SA)
            current_metrics, current_obj = evaluate_objective(
                state['current_data'], objective_func
            )
            
            # Create a combined metrics dictionary with proper prefixes
            all_metrics = {}
            
            # Add BEST metrics with "Best " prefix
            all_metrics['Best Objective'] = state['best_objective']
            for metric_name, metric_value in best_metrics.items():
                all_metrics[f'Best {metric_name}'] = metric_value
            
            # Add CURRENT metrics with "Current " prefix
            all_metrics['Current Objective'] = state['current_objective']
            for metric_name, metric_value in current_metrics.items():
                all_metrics[f'Current {metric_name}'] = metric_value
            
            # Buffer all metrics for this step
            for metric_name, metric_value in all_metrics.items():
                db_buffer.append((state['replica_id'], state['step'], metric_name, metric_value))
            
            # Flush buffer if it reaches buffer_size
            if len(db_buffer) >= db_buffer_size * len(all_metrics):
                db_writer.insert_metrics_batch(db_buffer)
                db_buffer = []
    
    # Return state (already in dict format)
    return state


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
