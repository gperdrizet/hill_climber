"""Hill climbing optimization with replica exchange."""

import numpy as np
import pandas as pd
import pickle
import time
import os
from typing import Callable, Optional, Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

from .optimizer_state import OptimizerState
from .climber_functions import perturb_vectors, evaluate_objective
from .plotting_functions import plot_results as plot_results_func
from .replica_exchange import (
    TemperatureLadder, ExchangeScheduler, ExchangeStatistics,
    should_exchange
)
from .replica_worker import run_replica_steps


class HillClimber:
    """Hill climbing optimizer with replica exchange.
    
    This optimizer uses parallel tempering (replica exchange) to improve
    global optimization. Multiple replicas run at different temperatures,
    periodically exchanging configurations to enhance exploration and
    exploitation.
    
    Args:
        data: Input data as numpy array (N, M) or pandas DataFrame with M columns
        objective_func: Function taking M column arrays, returns (metrics_dict, objective_value)
        max_time: Maximum runtime in minutes
        perturb_fraction: Fraction of data points to perturb each step
        temperature: Base temperature (will be used as T_min for ladder)
        cooling_rate: Temperature decay rate per step
        mode: 'maximize', 'minimize', or 'target'
        target_value: Target value (only used if mode='target')
        checkpoint_file: Path to save checkpoints
        save_interval: Checkpoint save interval in seconds
        plot_progress: Plot results every N minutes (None to disable)
        plot_metrics: List of metric names to plot (None to plot all metrics)
        step_spread: Standard deviation for perturbation distribution
        n_replicas: Number of replicas for parallel tempering
        T_max: Maximum temperature (hottest replica)
        exchange_interval: Steps between exchange attempts
        temperature_scheme: 'geometric' or 'linear' temperature spacing
        exchange_strategy: 'even_odd', 'random', or 'all_neighbors'
        n_workers: Number of worker processes (None = n_replicas, 0 = sequential)
    """
    
    def __init__(
        self,
        data,
        objective_func: Callable,
        max_time: float = 30,
        perturb_fraction: float = 0.05,
        temperature: float = 1000,
        cooling_rate: float = 0.000001,
        mode: str = 'maximize',
        target_value: Optional[float] = None,
        checkpoint_file: Optional[str] = None,
        save_interval: float = 60,
        plot_progress: Optional[float] = None,
        plot_metrics: Optional[List[str]] = None,
        step_spread: float = 1.0,
        n_replicas: Optional[int] = None,
        T_max: Optional[float] = None,
        exchange_interval: int = 100,
        temperature_scheme: str = 'geometric',
        exchange_strategy: str = 'even_odd',
        n_workers: Optional[int] = None
    ):
        # Convert data to numpy if needed
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.column_names = list(data.columns)
            self.is_dataframe = True
        else:
            self.data = np.array(data)
            self.column_names = [f'col_{i}' for i in range(self.data.shape[1])]
            self.is_dataframe = False
        
        # Store bounds for boundary reflection
        self.bounds = (np.min(self.data, axis=0), np.max(self.data, axis=0))
        
        self.objective_func = objective_func
        self.max_time = max_time
        self.perturb_fraction = perturb_fraction
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.mode = mode
        self.target_value = target_value
        self.checkpoint_file = checkpoint_file
        self.save_interval = save_interval
        self.plot_progress = plot_progress
        self.plot_metrics = plot_metrics
        self.step_spread = step_spread
        
        # Replica exchange parameters
        self.n_replicas = n_replicas or 4  # Default to 4 replicas
        self.T_min = temperature
        self.T_max = T_max or (temperature * 10)
        self.exchange_interval = exchange_interval
        self.temperature_scheme = temperature_scheme
        self.exchange_strategy = exchange_strategy
        
        # Parallel processing parameters
        # None = use n_replicas workers, 0 = sequential mode, >0 = specified number
        if n_workers is None:
            self.n_workers = self.n_replicas
        elif n_workers == 0:
            self.n_workers = 0  # Sequential mode
        else:
            self.n_workers = min(n_workers, cpu_count())
        
        # Will be initialized in climb()
        self.replicas: List[OptimizerState] = []
        self.temperature_ladder: Optional[TemperatureLadder] = None
        self.exchange_stats: Optional[ExchangeStatistics] = None
        self.last_save_time: Optional[float] = None
        self.last_plot_time: Optional[float] = None


    def climb(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run replica exchange optimization.
        
        Returns:
            Tuple of (best_data, steps_df) where:
                best_data: Best configuration found across all replicas
                steps_df: DataFrame with optimization history from best replica
        """
        mode_str = f"parallel ({self.n_workers} workers)" if self.n_workers > 0 else "sequential"
        print(f"Starting replica exchange with {self.n_replicas} replicas ({mode_str})...")
        
        # Initialize temperature ladder
        if self.temperature_scheme == 'geometric':
            self.temperature_ladder = TemperatureLadder.geometric(
                self.n_replicas, self.T_min, self.T_max
            )
        else:
            self.temperature_ladder = TemperatureLadder.linear(
                self.n_replicas, self.T_min, self.T_max
            )
        
        print(f"Temperature ladder: {self.temperature_ladder.temperatures}")
        
        # Initialize replicas
        self._initialize_replicas()
        
        # Initialize exchange scheduler and statistics
        scheduler = ExchangeScheduler(self.n_replicas, self.exchange_strategy)
        self.exchange_stats = ExchangeStatistics(self.n_replicas)
        
        # Choose execution path
        if self.n_workers > 0:
            return self._climb_parallel(scheduler)
        else:
            return self._climb_sequential(scheduler)
    
    def _climb_sequential(self, scheduler: ExchangeScheduler) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run optimization sequentially (original single-threaded approach)."""
        # Main optimization loop
        start_time = time.time()
        self.last_save_time = start_time
        self.last_plot_time = start_time
        
        while (time.time() - start_time) < (self.max_time * 60):
            
            # Each replica takes a step
            for replica in self.replicas:
                self._step_replica(replica)
            
            # Attempt exchanges periodically
            if self.replicas[0].step % self.exchange_interval == 0:
                self._exchange_round(scheduler)
            
            # Checkpointing
            if self.checkpoint_file and (time.time() - self.last_save_time >= self.save_interval):
                self.save_checkpoint(self.checkpoint_file)
                self.last_save_time = time.time()
            
            # Progress plotting
            if self.plot_progress and (time.time() - self.last_plot_time >= self.plot_progress * 60):
                self._plot_progress()
                self.last_plot_time = time.time()
        
        return self._finalize_results()
    
    def _climb_parallel(self, scheduler: ExchangeScheduler) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run optimization with parallel workers."""
        start_time = time.time()
        self.last_save_time = start_time
        self.last_plot_time = start_time
        
        # Create worker pool
        with Pool(processes=self.n_workers) as pool:
            while (time.time() - start_time) < (self.max_time * 60):
                
                # Run batch of steps in parallel
                self._parallel_step_batch(pool, self.exchange_interval)
                
                # Attempt exchanges
                self._exchange_round(scheduler)
                
                # Checkpointing
                if self.checkpoint_file and (time.time() - self.last_save_time >= self.save_interval):
                    self.save_checkpoint(self.checkpoint_file)
                    self.last_save_time = time.time()
                
                # Progress plotting
                if self.plot_progress and (time.time() - self.last_plot_time >= self.plot_progress * 60):
                    self._plot_progress()
                    self.last_plot_time = time.time()
        
        return self._finalize_results()
    
    def _parallel_step_batch(self, pool: Pool, n_steps: int):
        """Execute n_steps for all replicas in parallel."""
        # Serialize current replica states
        state_dicts = [self._serialize_state(r) for r in self.replicas]
        
        # Create partial function with fixed parameters
        worker_func = partial(
            run_replica_steps,
            objective_func=self.objective_func,
            bounds=self.bounds,
            column_names=self.column_names,
            n_steps=n_steps,
            mode=self.mode,
            target_value=self.target_value
        )
        
        # Execute in parallel
        updated_states = pool.map(worker_func, state_dicts)
        
        # Deserialize results back into replica objects
        for i, state_dict in enumerate(updated_states):
            self.replicas[i] = OptimizerState(**state_dict)
    
    def _finalize_results(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Complete optimization and return results."""
        # Final checkpoint and plot
        if self.checkpoint_file:
            self.save_checkpoint(self.checkpoint_file)
        if self.plot_progress:
            self._plot_progress(force=True)
        
        # Return results from best replica
        best_replica = self._get_best_replica()
        print(f"\nBest result from replica {best_replica.replica_id} "
              f"(T={best_replica.temperature:.1f})")
        print(f"Exchange acceptance rate: {self.exchange_stats.get_overall_acceptance_rate():.2%}")
        
        # Convert to DataFrame if input was DataFrame
        if self.is_dataframe:
            best_data_output = pd.DataFrame(best_replica.best_data, columns=self.column_names)
        else:
            best_data_output = best_replica.best_data
        
        return best_data_output, best_replica.get_history_dataframe()
    
    def _initialize_replicas(self):
        """Initialize all replica states."""
        hyperparams = {
            'max_time': self.max_time,
            'perturb_fraction': self.perturb_fraction,
            'cooling_rate': self.cooling_rate,
            'mode': self.mode,
            'target_value': self.target_value,
            'step_spread': self.step_spread
        }
        
        # Evaluate initial objective
        metrics, objective = evaluate_objective(
            self.data, self.objective_func, self.column_names
        )
        
        self.replicas = []
        for i, temp in enumerate(self.temperature_ladder.temperatures):
            state = OptimizerState(
                replica_id=i,
                temperature=temp,
                current_data=self.data.copy(),
                current_objective=objective,
                best_data=self.data.copy(),
                best_objective=objective,
                original_data=self.data.copy(),
                hyperparameters=hyperparams.copy()
            )
            state.record_step(metrics, objective)
            self.replicas.append(state)
    
    def _step_replica(self, replica: OptimizerState):
        """Perform one optimization step for a replica."""
        # Perturb data (already includes boundary reflection)
        perturbed = perturb_vectors(
            replica.current_data,
            self.perturb_fraction,
            self.bounds,
            self.step_spread
        )
        
        # Evaluate
        metrics, objective = evaluate_objective(
            perturbed, self.objective_func, self.column_names
        )
        
        # Acceptance criterion (simulated annealing)
        accept = self._should_accept(
            objective, replica.current_objective, replica.temperature
        )
        
        if accept:
            replica.record_improvement(perturbed, objective, metrics)
        else:
            replica.record_step(metrics, replica.current_objective)
        
        # Cool temperature
        replica.temperature *= (1 - self.cooling_rate)
    
    def _should_accept(self, new_obj: float, current_obj: float, temp: float) -> bool:
        """Determine if new state should be accepted (simulated annealing)."""
        if self.mode == 'maximize':
            delta = new_obj - current_obj
        elif self.mode == 'minimize':
            delta = current_obj - new_obj
        else:  # target mode
            current_dist = abs(current_obj - self.target_value)
            new_dist = abs(new_obj - self.target_value)
            delta = current_dist - new_dist
        
        if delta > 0:
            return True
        else:
            prob = np.exp(delta / temp) if temp > 0 else 0
            return np.random.random() < prob
    
    def _exchange_round(self, scheduler: ExchangeScheduler):
        """Perform one round of replica exchanges."""
        pairs = scheduler.get_pairs()
        
        for i, j in pairs:
            replica_i = self.replicas[i]
            replica_j = self.replicas[j]
            
            # Attempt exchange
            accepted = should_exchange(
                replica_i.current_objective,
                replica_j.current_objective,
                replica_i.temperature,
                replica_j.temperature,
                self.mode
            )
            
            if accepted:
                # Swap configurations (not temperatures!)
                replica_i.current_data, replica_j.current_data = \
                    replica_j.current_data.copy(), replica_i.current_data.copy()
                replica_i.current_objective, replica_j.current_objective = \
                    replica_j.current_objective, replica_i.current_objective
            
            # Record statistics
            replica_i.record_exchange(j, accepted)
            replica_j.record_exchange(i, accepted)
            self.exchange_stats.record_attempt(i, j, accepted)
        
        if pairs:
            self.exchange_stats.round_count += 1
    
    def _get_best_replica(self) -> OptimizerState:
        """Find replica with best objective value."""
        if self.mode == 'maximize':
            return max(self.replicas, key=lambda r: r.best_objective)
        elif self.mode == 'minimize':
            return min(self.replicas, key=lambda r: r.best_objective)
        else:  # target mode
            return min(self.replicas, 
                      key=lambda r: abs(r.best_objective - self.target_value))
    
    def _plot_progress(self, force: bool = False):
        """Plot current progress of best replica."""
        best_replica = self._get_best_replica()
        
        if not best_replica.metrics_history:
            elapsed = (time.time() - best_replica.start_time) / 60
            if elapsed < 60:
                time_str = f"{int(elapsed)} minutes"
            else:
                time_str = f"{elapsed/60:.1f} hours"
            print(f"No accepted steps yet (elapsed: {time_str})")
            return
        
        # Clear previous plots
        plt.close('all')
        
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass
        
        # Create results structure for plotting
        if self.is_dataframe:
            best_data_output = pd.DataFrame(best_replica.best_data, columns=self.column_names)
        else:
            best_data_output = best_replica.best_data
        
        results = (best_data_output, best_replica.get_history_dataframe())
        self.plot_results(results)
    
    def plot_results(self, results, plot_type: str = 'scatter', 
                     metrics: Optional[List[str]] = None):
        """Plot optimization results.
        
        Args:
            results: Tuple of (best_data, steps_df)
            plot_type: 'scatter' or 'histogram'
            metrics: List of metric names to plot (overrides plot_metrics from __init__)
        """
        # Use provided metrics, or fall back to instance plot_metrics
        metrics_to_plot = metrics if metrics is not None else self.plot_metrics
        
        # Wrap single result for compatibility with plot_results function
        if isinstance(results, tuple) and len(results) == 2:
            best_data, steps_df = results
            
            # Convert input data for plotting
            if self.is_dataframe:
                input_data = pd.DataFrame(self.data, columns=self.column_names)
            else:
                input_data = self.data
            
            wrapped_results = {
                'input_data': input_data,
                'results': [(self.data, best_data, steps_df)]
            }
        else:
            wrapped_results = results
        
        plot_results_func(wrapped_results, plot_type, metrics_to_plot)
    
    def save_checkpoint(self, filepath: str):
        """Save current state to checkpoint file."""
        checkpoint = {
            'replicas': [self._serialize_state(r) for r in self.replicas],
            'temperature_ladder': self.temperature_ladder.temperatures.tolist(),
            'exchange_stats': {
                'attempts': self.exchange_stats.attempts.tolist(),
                'acceptances': self.exchange_stats.acceptances.tolist(),
                'round_count': self.exchange_stats.round_count
            },
            'elapsed_time': time.time() - self.replicas[0].start_time,
            'hyperparameters': self.replicas[0].hyperparameters,
            'is_dataframe': self.is_dataframe,
            'column_names': self.column_names,
            'bounds': self.bounds,
            'plot_metrics': self.plot_metrics,
            'n_workers': self.n_workers
        }
        
        # Create checkpoint directory if needed
        checkpoint_dir = os.path.dirname(filepath)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved: {filepath}")
    
    @staticmethod
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
    
    @classmethod
    def load_checkpoint(cls, filepath: str, objective_func: Callable):
        """Load optimization state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            objective_func: Objective function (must match original)
            
        Returns:
            HillClimber instance with restored state
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Reconstruct climber
        first_replica = checkpoint['replicas'][0]
        hyperparams = first_replica['hyperparameters']
        
        # Create data in proper format
        data = first_replica['original_data']
        if checkpoint.get('is_dataframe', False):
            data = pd.DataFrame(data, columns=checkpoint['column_names'])
        
        climber = cls(
            data=data,
            objective_func=objective_func,
            max_time=hyperparams['max_time'],
            perturb_fraction=hyperparams['perturb_fraction'],
            temperature=hyperparams.get('temperature', 1000),
            cooling_rate=hyperparams['cooling_rate'],
            mode=hyperparams['mode'],
            target_value=hyperparams.get('target_value'),
            step_spread=hyperparams['step_spread'],
            n_replicas=len(checkpoint['replicas']),
            plot_metrics=checkpoint.get('plot_metrics'),
            n_workers=checkpoint.get('n_workers')
        )
        
        # Restore states
        climber.replicas = [
            OptimizerState(**state_dict)
            for state_dict in checkpoint['replicas']
        ]
        
        # Restore temperature ladder
        climber.temperature_ladder = TemperatureLadder(
            temperatures=np.array(checkpoint['temperature_ladder'])
        )
        
        # Restore exchange statistics
        stats_data = checkpoint['exchange_stats']
        climber.exchange_stats = ExchangeStatistics(len(climber.replicas))
        climber.exchange_stats.attempts = np.array(stats_data['attempts'])
        climber.exchange_stats.acceptances = np.array(stats_data['acceptances'])
        climber.exchange_stats.round_count = stats_data['round_count']
        
        # Restore bounds
        climber.bounds = checkpoint.get('bounds', (np.min(climber.data, axis=0), np.max(climber.data, axis=0)))
        
        print(f"Resumed from checkpoint: {checkpoint['elapsed_time']/60:.1f} minutes elapsed")
        
        return climber
