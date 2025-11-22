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
    target_value: float = None
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
    
    Returns:
        Updated serialized state dictionary
    """
    # Deserialize state
    state = OptimizerState(**state_dict)
    
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
        
        # Acceptance criterion (simulated annealing)
        accept = _should_accept(
            objective, state.current_objective, state.temperature,
            mode, target_value
        )
        
        if accept:
            state.record_improvement(perturbed, objective, metrics)
        # Note: We only record accepted steps to avoid misleading history
        # where rejected steps would show the old objective with a new step number
        
        # Cool temperature
        cooling_rate = state.hyperparameters['cooling_rate']
        state.temperature *= (1 - cooling_rate)
    
    # Serialize and return
    return _serialize_state(state)


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
