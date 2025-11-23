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
        checkpoint_file: Path to save checkpoints (default: None, no checkpointing)
        plot_metrics: List of metric names to plot (None to plot all metrics)
        plot_type: Type of plot for progress snapshots ('scatter' or 'histogram')
        show_progress: Show progress plots during optimization (default: True)
        verbose: Print progress messages (default: False)
        step_spread: Perturbation spread as fraction of input range (default: 0.01 = 1%)
        n_replicas: Number of replicas for parallel tempering (default: 4)
        T_max: Maximum temperature for hottest replica (default: 10 * temperature)
        exchange_interval: Steps between exchange attempts
        temperature_scheme: 'geometric' or 'linear' temperature spacing
        exchange_strategy: 'even_odd', 'random', or 'all_neighbors'
        n_workers: Number of worker processes (None = n_replicas, 0 = sequential)
        db_enabled: Enable database logging for dashboard (default: False)
        db_path: Path to SQLite database file (default: 'data/hill_climber_progress.db')
        db_step_interval: Collect metrics every Nth step (default: exchange_interval // 1000)
        db_buffer_size: Number of pooled steps before database write (default: 10)
        checkpoint_interval: Batches between checkpoint saves (default: 1, i.e., every batch)
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
        plot_metrics: Optional[List[str]] = None,
        plot_type: str = 'scatter',
        show_progress: bool = True,
        verbose: bool = False,
        step_spread: float = 0.01,
        n_replicas: int = 4,
        T_max: Optional[float] = None,
        exchange_interval: int = 1000,
        temperature_scheme: str = 'geometric',
        exchange_strategy: str = 'even_odd',
        n_workers: Optional[int] = None,
        db_enabled: bool = False,
        db_path: Optional[str] = None,
        db_step_interval: Optional[int] = None,
        db_buffer_size: int = 10,
        checkpoint_interval: int = 1
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
        
        # Calculate absolute step_spread from fraction of data range
        data_range = self.bounds[1] - self.bounds[0]
        self.step_spread_absolute = step_spread * np.mean(data_range)
        
        # Validate mode
        if mode not in ['maximize', 'minimize', 'target']:
            raise ValueError(f"mode must be 'maximize', 'minimize', or 'target', got '{mode}'")
        
        # Validate target_value when mode is 'target'
        if mode == 'target' and target_value is None:
            raise ValueError("target_value must be specified when mode='target'")
        
        self.objective_func = objective_func
        self.max_time = max_time
        self.perturb_fraction = perturb_fraction
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.mode = mode
        self.target_value = target_value
        self.checkpoint_file = checkpoint_file
        self.plot_metrics = plot_metrics
        self.plot_type = plot_type
        self.show_progress = show_progress
        self.verbose = verbose
        self.step_spread = step_spread
        
        # Replica exchange parameters
        self.n_replicas = n_replicas
        self.T_min = temperature
        self.T_max = T_max or (temperature * 10)
        self.exchange_interval = exchange_interval
        self.temperature_scheme = temperature_scheme
        self.exchange_strategy = exchange_strategy
        
        # Database parameters
        self.db_enabled = db_enabled
        self.checkpoint_interval = checkpoint_interval
        
        if db_enabled:
            # Set database path (default to data/ directory)
            if db_path is None:
                self.db_path = 'data/hill_climber_progress.db'
            else:
                self.db_path = db_path
            
            # Set step interval (default: 0.1% of exchange interval)
            self.db_step_interval = db_step_interval if db_step_interval is not None else max(1, exchange_interval // 1000)
            self.db_buffer_size = db_buffer_size
            
            # Import database module only if enabled
            from .database import DatabaseWriter
            self.db_writer = DatabaseWriter(self.db_path)
        else:
            self.db_path = None
            self.db_step_interval = None
            self.db_buffer_size = None
            self.db_writer = None
        
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
        self.batch_counter = 0  # Track batches for checkpoint_interval


    def climb(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run replica exchange optimization.
        
        Returns:
            Tuple of (best_data, steps_df) where:
                best_data: Best configuration found across all replicas
                steps_df: DataFrame with optimization history from best replica
        """
        mode_str = f"parallel ({self.n_workers} workers)" if self.n_workers > 0 else "sequential"
        if self.verbose:
            print(f"Starting replica exchange with {self.n_replicas} replicas ({mode_str})...")
        
        # Initialize database if enabled
        if self.db_enabled:
            self._initialize_database()
        
        # Initialize temperature ladder
        if self.temperature_scheme == 'geometric':
            self.temperature_ladder = TemperatureLadder.geometric(
                self.n_replicas, self.T_min, self.T_max
            )
        else:
            self.temperature_ladder = TemperatureLadder.linear(
                self.n_replicas, self.T_min, self.T_max
            )
        
        if self.verbose:
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
        
        while (time.time() - start_time) < (self.max_time * 60):
            
            # Each replica takes a step
            for replica in self.replicas:
                self._step_replica(replica)
            
            # Attempt exchanges periodically
            if self.replicas[0].step % self.exchange_interval == 0:
                self._exchange_round(scheduler)
                
                # Increment batch counter
                self.batch_counter += 1
                
                # Checkpoint after every checkpoint_interval batches
                if self.checkpoint_file and (self.batch_counter % self.checkpoint_interval == 0):
                    self.save_checkpoint(self.checkpoint_file)
        
        return self._finalize_results()
    
    def _climb_parallel(self, scheduler: ExchangeScheduler) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run optimization with parallel workers."""
        start_time = time.time()
        
        # Create worker pool
        with Pool(processes=self.n_workers) as pool:
            while (time.time() - start_time) < (self.max_time * 60):
                
                # Run batch of steps in parallel
                self._parallel_step_batch(pool, self.exchange_interval)
                
                # Attempt exchanges
                self._exchange_round(scheduler)
                
                # Increment batch counter
                self.batch_counter += 1
                
                # Checkpoint after every checkpoint_interval batches
                if self.checkpoint_file and (self.batch_counter % self.checkpoint_interval == 0):
                    self.save_checkpoint(self.checkpoint_file)
        
        return self._finalize_results()
    
    def _parallel_step_batch(self, pool: Pool, n_steps: int):
        """Execute n_steps for all replicas in parallel."""
        # Serialize current replica states
        state_dicts = [self._serialize_state(r) for r in self.replicas]
        
        # Prepare database config if enabled
        db_config = None
        if self.db_enabled:
            db_config = {
                'enabled': True,
                'path': self.db_path,
                'step_interval': self.db_step_interval,
                'buffer_size': self.db_buffer_size
            }
        
        # Create partial function with fixed parameters
        worker_func = partial(
            run_replica_steps,
            objective_func=self.objective_func,
            bounds=self.bounds,
            n_steps=n_steps,
            mode=self.mode,
            target_value=self.target_value,
            db_config=db_config
        )
        
        # Execute in parallel
        updated_states = pool.map(worker_func, state_dicts)
        
        # Collect database buffers and update replicas
        all_db_buffers = []
        for i, state_dict in enumerate(updated_states):
            # Extract and collect database buffer if present
            if 'db_buffer' in state_dict:
                all_db_buffers.extend(state_dict.pop('db_buffer'))
            
            # Preserve temperature_history before updating
            temp_history = self.replicas[i].temperature_history
            self.replicas[i] = OptimizerState(**state_dict)
            self.replicas[i].temperature_history = temp_history
        
        # Flush all collected database buffers to database
        if self.db_enabled and all_db_buffers:
            self.db_writer.insert_metrics_batch(all_db_buffers)
        
        # Update replica status in database
        if self.db_enabled:
            for replica in self.replicas:
                self.db_writer.update_replica_status(
                    replica_id=replica.replica_id,
                    step=replica.step,
                    temperature=replica.temperature,
                    best_objective=replica.best_objective,
                    current_objective=replica.current_objective
                )
    
    def _finalize_results(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Complete optimization and return results."""
        # Final checkpoint
        if self.checkpoint_file:
            self.save_checkpoint(self.checkpoint_file)
        
        # Return results from best replica
        best_replica = self._get_best_replica()
        
        if self.verbose:
            print(f"\nBest result from replica {best_replica.replica_id} "
                  f"(T={best_replica.temperature:.1f})")
            print(f"Exchange acceptance rate: {self.exchange_stats.get_overall_acceptance_rate():.2%}")
        
        # Convert to DataFrame if input was DataFrame
        if self.is_dataframe:
            best_data_output = pd.DataFrame(best_replica.best_data, columns=self.column_names)
        else:
            best_data_output = best_replica.best_data
        
        return best_data_output, best_replica.get_history_dataframe()
    
    def _initialize_database(self):
        """Initialize database schema and insert run metadata."""
        if not self.db_enabled or not self.db_writer:
            return
        
        # Create database directory if needed
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Drop existing tables and create fresh schema
        self.db_writer.initialize_schema(drop_existing=True)
        
        # Insert run metadata
        hyperparameters = {
            'max_time': self.max_time,
            'perturb_fraction': self.perturb_fraction,
            'temperature': self.temperature,
            'cooling_rate': self.cooling_rate,
            'mode': self.mode,
            'target_value': self.target_value,
            'step_spread': self.step_spread,
            'T_min': self.T_min,
            'T_max': self.T_max,
            'temperature_scheme': self.temperature_scheme,
            'exchange_strategy': self.exchange_strategy
        }
        
        self.db_writer.insert_run_metadata(
            n_replicas=self.n_replicas,
            exchange_interval=self.exchange_interval,
            db_step_interval=self.db_step_interval,
            db_buffer_size=self.db_buffer_size,
            hyperparameters=hyperparameters
        )
        
        if self.verbose:
            print(f"Database initialized: {self.db_path}")
            print(f"  Step interval: {self.db_step_interval} (collecting every {self.db_step_interval}th step)")
            print(f"  Buffer size: {self.db_buffer_size} (writing every {self.db_buffer_size} collected steps)")
    
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
            self.data, self.objective_func
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
            self.step_spread_absolute
        )
        
        # Evaluate
        metrics, objective = evaluate_objective(
            perturbed, self.objective_func
        )
        
        # Acceptance criterion (simulated annealing)
        accept = self._should_accept(
            objective, replica.current_objective, replica.temperature
        )
        
        if accept:
            replica.record_improvement(perturbed, objective, metrics)
        # Note: We only record accepted steps to avoid misleading history
        # where rejected steps would show the old objective with a new step number
        
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
        
        # Track exchanges for database logging
        db_exchanges = []
        
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
                # Swap temperatures (not data!) to keep each replica's history continuous
                old_temp_i = replica_i.temperature
                old_temp_j = replica_j.temperature
                
                # Record temperature changes with current step number
                current_step = replica_i.step  # Use step from replica i
                replica_i.record_temperature_change(old_temp_j, current_step)
                replica_j.record_temperature_change(old_temp_i, replica_j.step)
                
                # Collect for database logging
                if self.db_enabled:
                    db_exchanges.append((current_step, i, old_temp_j))
                    db_exchanges.append((replica_j.step, j, old_temp_i))
            
            # Record statistics
            replica_i.record_exchange(j, accepted)
            replica_j.record_exchange(i, accepted)
            self.exchange_stats.record_attempt(i, j, accepted)
        
        if pairs:
            self.exchange_stats.round_count += 1
        
        # Log exchanges to database
        if self.db_enabled and db_exchanges:
            self.db_writer.insert_temperature_exchanges(db_exchanges)
    
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
        """Plot current progress of all replicas."""
        # Check if any replica has accepted steps
        has_data = any(replica.metrics_history for replica in self.replicas)
        
        if not has_data:
            best_replica = self._get_best_replica()
            elapsed = (time.time() - best_replica.start_time) / 60
            if elapsed < 60:
                time_str = f"{int(elapsed)} minutes"
            else:
                time_str = f"{elapsed/60:.1f} hours"
            if self.verbose:
                print(f"No accepted steps yet (elapsed: {time_str})")
            return
        
        # Clear previous plots
        plt.close('all')
        
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass
        
        # Create results structure for all replicas
        results_list = []
        for replica in self.replicas:
            if replica.metrics_history:  # Only include replicas with data
                if self.is_dataframe:
                    replica_data = pd.DataFrame(replica.best_data, columns=self.column_names)
                else:
                    replica_data = replica.best_data
                
                # Include temperature_history as first element of tuple
                results_list.append((replica.temperature_history, replica_data, replica.get_history_dataframe()))
        
        if results_list:
            self.plot_results(results_list, plot_type=self.plot_type)
    
    def plot_results(self, results, plot_type: str = 'scatter', 
                     metrics: Optional[List[str]] = None, all_replicas: bool = False):
        """Plot optimization results.
        
        Args:
            results: Tuple of (best_data, steps_df) or list of such tuples
            plot_type: 'scatter' or 'histogram'
            metrics: List of metric names to plot (overrides plot_metrics from __init__)
            all_replicas: If True, plot all replicas instead of just the results passed in
        """
        # Use provided metrics, or fall back to instance plot_metrics
        metrics_to_plot = metrics if metrics is not None else self.plot_metrics
        
        # If all_replicas is True, gather results from all replicas
        if all_replicas:
            all_results = []
            for replica in self.replicas:
                if self.is_dataframe:
                    replica_data = pd.DataFrame(replica.best_data, columns=self.column_names)
                else:
                    replica_data = replica.best_data
                all_results.append((replica_data, replica.get_history_dataframe()))
            plot_results_func(all_results, plot_type, metrics_to_plot, self.exchange_interval)
        else:
            plot_results_func(results, plot_type, metrics_to_plot, self.exchange_interval)
    
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
            'plot_type': self.plot_type,
            'verbose': self.verbose,
            'n_workers': self.n_workers
        }
        
        # Create checkpoint directory if needed
        checkpoint_dir = os.path.dirname(filepath)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if self.verbose:
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
            'temperature_history': state.temperature_history,
            'exchange_attempts': state.exchange_attempts,
            'exchange_acceptances': state.exchange_acceptances,
            'partner_history': state.partner_history,
            'original_data': state.original_data,
            'hyperparameters': state.hyperparameters,
            'start_time': state.start_time
        }
    
    @classmethod
    def load_checkpoint(cls, filepath: str, objective_func: Callable, reset_temperatures: bool = False):
        """Load optimization state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            objective_func: Objective function (must match original)
            reset_temperatures: If True, reset replica temperatures to original ladder values
                              If False (default), continue from saved temperatures
            
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
            plot_type=checkpoint.get('plot_type', 'scatter'),
            verbose=checkpoint.get('verbose', False),
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
        
        # Get elapsed time from checkpoint
        elapsed_seconds = checkpoint['elapsed_time']
        
        # Reset temperatures if requested
        if reset_temperatures:
            for i, replica in enumerate(climber.replicas):
                replica.temperature = climber.temperature_ladder.temperatures[i]
            if climber.verbose:
                print(f"Resumed from checkpoint with reset temperatures: {elapsed_seconds/60:.1f} minutes elapsed")
        else:
            if climber.verbose:
                print(f"Resumed from checkpoint: {elapsed_seconds/60:.1f} minutes elapsed")
        
        # Restore exchange statistics
        stats_data = checkpoint['exchange_stats']
        climber.exchange_stats = ExchangeStatistics(len(climber.replicas))
        climber.exchange_stats.attempts = np.array(stats_data['attempts'])
        climber.exchange_stats.acceptances = np.array(stats_data['acceptances'])
        climber.exchange_stats.round_count = stats_data['round_count']
        
        # Restore bounds
        climber.bounds = checkpoint.get('bounds', (np.min(climber.data, axis=0), np.max(climber.data, axis=0)))
        
        # Adjust replica start times to account for elapsed time
        # When resuming, we want elapsed time calculations to continue from where they left off
        # So we set start_time = current_time - elapsed_time
        current_time = time.time()
        adjusted_start_time = current_time - elapsed_seconds
        
        for replica in climber.replicas:
            replica.start_time = adjusted_start_time
        
        return climber
