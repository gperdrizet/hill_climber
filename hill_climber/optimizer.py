"""Hill climbing optimization with replica exchange."""

import os
import pickle
import time
from typing import Callable, Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import Pool as PoolType

from .optimizer_state import create_replica_state, record_temperature_change, record_exchange, get_history_dataframe
from .climber_functions import perturb_vectors, evaluate_objective
from .replica_exchange import (
    TemperatureLadder, ExchangeScheduler, should_exchange
)
from .replica_worker import run_replica_steps
from .config import (
    OptimizerConfig,
    DEFAULT_T_MIN,
    DEFAULT_T_MAX_MULTIPLIER,
    DEFAULT_COOLING_RATE,
    DEFAULT_STEP_SPREAD,
    DEFAULT_PERTURB_FRACTION,
    DEFAULT_N_REPLICAS,
    DEFAULT_EXCHANGE_INTERVAL,
    DEFAULT_TEMPERATURE_SCHEME,
    DEFAULT_EXCHANGE_STRATEGY,
    DEFAULT_MAX_TIME,
    DEFAULT_MODE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DB_PATH,
    DB_STEP_INTERVAL_DIVISOR,
    DEFAULT_COLUMN_PREFIX,
)


class HillClimber:
    """Hill climbing optimizer with replica exchange.
    
    This optimizer uses parallel tempering (replica exchange) to improve
    global optimization. Multiple replicas run at different temperatures,
    periodically exchanging configurations to enhance exploration and
    exploitation.
    
    Args:
        data: Input data as numpy array (N, M) or pandas DataFrame with M columns
        objective_func: Function taking M column arrays, returns (metrics_dict, objective_value)
        mode: 'maximize', 'minimize', or 'target'
        target_value: Target value (only used if mode='target')
        max_time: Maximum runtime in minutes
        step_spread: Perturbation spread as fraction of input range (default: 0.01 = 1%). Step 
            values are sampled from a gaussian distribution with mean 0 and standard deviation = 
            input range * step_spread
        perturb_fraction: Fraction of data points to perturb each step
        n_replicas: Number of replicas for parallel tempering (default: 4), setting to 1 runs
            simulated annealing without replica exchange
        T_min: Base temperature (will be used as T_min for ladder)
        T_max: Maximum temperature for hottest replica (default: 100 * temperature)
        cooling_rate: Temperature decay rate per successful step
        temperature_scheme: 'geometric' or 'linear' temperature spacing
        exchange_interval: Steps between exchange attempts
        exchange_strategy: 'even_odd', 'random', or 'all_neighbors'
        checkpoint_file: Path to save checkpoints (default: None, no checkpointing)
        checkpoint_interval: Batches between checkpoint saves (default: 1, i.e., every batch)
        db_enabled: Enable database logging for dashboard (default: False)
        db_path: Path to SQLite database file (default: 'data/hill_climber_progress.db')
        db_step_interval: Collect metrics every Nth step (default: exchange_interval // 10, or 1 if exchange_interval <= 10)
        verbose: Print progress messages (default: False)
        n_workers: Number of worker processes (default: n_replicas)
    """
    
    def __init__(
        self,
        data,
        objective_func: Callable,
        mode: str = DEFAULT_MODE,
        target_value: Optional[float] = None,
        max_time: float = DEFAULT_MAX_TIME,
        step_spread: float = DEFAULT_STEP_SPREAD,
        perturb_fraction: float = DEFAULT_PERTURB_FRACTION,
        n_replicas: int = DEFAULT_N_REPLICAS,
        T_min: float = DEFAULT_T_MIN,
        T_max: Optional[float] = None,
        cooling_rate: float = DEFAULT_COOLING_RATE,
        temperature_scheme: str = DEFAULT_TEMPERATURE_SCHEME,
        exchange_interval: int = DEFAULT_EXCHANGE_INTERVAL,
        exchange_strategy: str = DEFAULT_EXCHANGE_STRATEGY,
        checkpoint_file: Optional[str] = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        db_enabled: bool = False,
        db_path: Optional[str] = None,
        db_step_interval: Optional[int] = None,
        verbose: bool = False,
        n_workers: Optional[int] = None,
    ):
        """Initialize HillClimber optimizer."""
        
        # Create config object for validation and store validated parameters
        # This delegates all validation to OptimizerConfig.__post_init__
        config = OptimizerConfig(
            objective_func=objective_func,
            mode=mode,
            target_value=target_value,
            max_time=max_time,
            step_spread=step_spread,
            perturb_fraction=perturb_fraction,
            n_replicas=n_replicas,
            T_min=T_min,
            T_max=T_max,
            cooling_rate=cooling_rate,
            temperature_scheme=temperature_scheme,
            exchange_interval=exchange_interval,
            exchange_strategy=exchange_strategy,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=checkpoint_interval,
            db_enabled=db_enabled,
            db_path=db_path,
            db_step_interval=db_step_interval,
            verbose=verbose,
            n_workers=n_workers,
        )
        
        # Convert data to numpy if needed
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.column_names = list(data.columns)
            self.is_dataframe = True

        else:
            self.data = np.array(data)
            self.column_names = [f'{DEFAULT_COLUMN_PREFIX}{i}' for i in range(self.data.shape[1])]
            self.is_dataframe = False


        #### Attribute assignments from validated config #############################
        
        # Hill climbing run parameters
        self.objective_func = config.objective_func
        self.mode = config.mode
        self.target_value = config.target_value
        self.max_time = config.max_time
        self.step_spread = config.step_spread
        self.perturb_fraction = config.perturb_fraction
        self.temperature = config.T_min
        self.cooling_rate = config.cooling_rate
        self.checkpoint_file = config.checkpoint_file
        self.verbose = config.verbose
        
        # Replica exchange parameters
        self.n_replicas = config.n_replicas
        self.T_min = config.T_min
        self.exchange_interval = config.exchange_interval
        self.temperature_scheme = config.temperature_scheme
        self.exchange_strategy = config.exchange_strategy
        
        # Database parameters
        self.db_enabled = config.db_enabled
        self.checkpoint_interval = config.checkpoint_interval

        # Placeholders - will be initialized in climb()
        self.replicas: List[Dict] = []
        self.temperature_ladder: Optional[TemperatureLadder] = None

        # Batch counter for checkpointing
        self.batch_counter = 0


        #### Derived attributes ######################################################

        # Highest temperature for replica ladder (already validated and set in config)
        self.T_max = config.T_max

        # Bounds for boundary reflection
        self.bounds = (np.min(self.data, axis=0), np.max(self.data, axis=0))
        
        # Absolute step_spread from fraction of data range
        data_range = self.bounds[1] - self.bounds[0]
        self.step_spread_absolute = config.step_spread * np.mean(data_range)
        
        # Database settings (already validated and defaults set in config)
        if config.db_enabled:
            self.db_path = config.db_path
            self.db_step_interval = config.db_step_interval
            
            # Import database module only if enabled
            from .database import DatabaseWriter

            self.db_writer = DatabaseWriter(self.db_path)

        else:
            self.db_path = None
            self.db_step_interval = None
            self.db_writer = None
        
        # Parallel processing parameters (already validated in config)
        if config.n_workers is None:
            self.n_workers = self.n_replicas
        elif config.n_workers == 0:
            self.n_workers = 0
        elif config.n_workers > cpu_count() - 1:
            print(
                "Warning: Requested n_workers + main process exceeds available " +
                f"CPU cores ({cpu_count()}). Consider decreasing n_replicas and/or n_workers"
            )
            self.n_workers = config.n_workers
        elif config.n_workers > config.n_replicas:
            print('Requested workers exceed number of replicas; reducing n_workers to n_replicas.')
            self.n_workers = config.n_replicas
        else:
            # Normal case: use the specified n_workers
            self.n_workers = config.n_workers        


    def climb(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run replica exchange optimization.
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Tuple of (best_data, steps_df) where:
                - best_data: Best configuration found across all replicas
                - steps_df: DataFrame with optimization history from best replica
        """

        if self.verbose:
            print(f"Starting replica exchange with {self.n_replicas} replicas...")
        
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

        if self.verbose:
            print(f"Initialized {len(self.replicas)} replicas.")

        # Initialize exchange scheduler and statistics
        scheduler = ExchangeScheduler(self.n_replicas, self.exchange_strategy)
        
        # Do the run
        return self._climb_parallel(scheduler)

    
    def _climb_parallel(self, scheduler: ExchangeScheduler) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run optimization with parallel workers.
        
        Args:
            scheduler (ExchangeScheduler): Scheduler for replica exchange.
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Tuple of (best_data, steps_df) from best replica.
        """

        start_time = time.time()
        avg_batch_time = 0.0
        
        # Create worker pool
        with Pool(processes=self.n_workers) as pool:
            while (time.time() - start_time) < (self.max_time * 60 - avg_batch_time):
                batch_start = time.time()
                
                # Run batch of steps in parallel
                self._parallel_step_batch(pool, self.exchange_interval)
                
                # Attempt exchanges if we are optimizing multiple replicas
                if self.n_replicas > 1:
                    self._exchange_round(scheduler)
                
                # Checkpoint after every checkpoint_interval batches
                if self.checkpoint_file and (self.batch_counter % self.checkpoint_interval == 0):
                    self.save_checkpoint(self.checkpoint_file)
                
                # Update average batch time
                batch_time = time.time() - batch_start
                if self.batch_counter == 0:
                    avg_batch_time = batch_time
                else:
                    # Running average: new_avg = old_avg + (new_value - old_avg) / (n + 1)
                    avg_batch_time = avg_batch_time + (batch_time - avg_batch_time) / (self.batch_counter + 1)
                
                # Increment batch counter
                self.batch_counter += 1
        
        return self._finalize_results()
    

    def _parallel_step_batch(self, pool: PoolType, n_steps: int):
        """Execute n_steps for all replicas in parallel.
        
        Args:
            pool (PoolType): Multiprocessing pool for parallel execution.
            n_steps (int): Number of optimization steps to execute per replica.
        """

        # Serialize current replica states
        state_dicts = [self._serialize_state(r) for r in self.replicas]
        
        # Prepare database config if enabled
        db_config = None

        if self.db_enabled:
            db_config = {
                'enabled': True,
                'path': self.db_path,
                'step_interval': self.db_step_interval
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
            temp_history = self.replicas[i]['temperature_history']
            self.replicas[i] = state_dict
            self.replicas[i]['temperature_history'] = temp_history
        
        # Flush all collected database buffers to database
        if self.db_enabled and all_db_buffers:
            self.db_writer.insert_metrics_batch(all_db_buffers)
        
        # Update replica status in database
        if self.db_enabled:
            for replica in self.replicas:
                self.db_writer.update_replica_status(
                    replica_id=replica['replica_id'],
                    step=replica['step'],
                    total_iterations=replica['total_iterations'],
                    temperature=replica['temperature'],
                    best_objective=replica['best_objective'],
                    current_objective=replica['current_objective']
                )
    

    def _finalize_results(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Complete optimization and return results.
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Tuple of (best_data, steps_df) from best replica.
        """

        # Final checkpoint
        if self.checkpoint_file:
            self.save_checkpoint(self.checkpoint_file)
        
        # Return results from best replica
        best_replica = self._get_best_replica()
        
        if self.verbose:
            print(
                f"\nBest result from replica {best_replica['replica_id']} "
                f"(T={best_replica['temperature']:.1f})"
            )

        # Convert to DataFrame if input was DataFrame
        if self.is_dataframe:
            best_data_output = pd.DataFrame(best_replica['best_data'], columns=self.column_names)
        
        else:
            best_data_output = best_replica['best_data']
        
        return best_data_output, get_history_dataframe(best_replica)
    

    def _initialize_database(self):
        """Initialize database schema and insert run metadata.
        
        Creates database directory if needed, drops existing tables, creates fresh schema,
        and inserts run metadata.
        """

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
            hyperparameters=hyperparameters,
            checkpoint_file=self.checkpoint_file,
            objective_function_name=self.objective_func.__name__ if hasattr(self.objective_func, '__name__') else None,
            dataset_size=len(self.data)
        )
        
        if self.verbose:
            print(f"Database initialized: {self.db_path}")
            print(f"  Step interval: {self.db_step_interval} (collecting every {self.db_step_interval}th step)")
    

    def _initialize_replicas(self):
        """Initialize all replica states.
        
        Creates replica states with temperatures from the temperature ladder,
        evaluates initial objective, and records initial metrics.
        """

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
            state = create_replica_state(
                replica_id=i,
                temperature=temp,
                current_data=self.data.copy(),
                current_objective=objective,
                best_data=self.data.copy(),
                best_objective=objective,
                original_data=self.data.copy(),
                hyperparameters=hyperparams.copy()
            )

            # Record initial metrics (including objective value)
            initial_metrics = metrics.copy()
            initial_metrics['Objective value'] = objective
            state['metrics_history'].append(initial_metrics)
            state['step'] += 1
            self.replicas.append(state)
    

    def _step_replica(self, replica: Dict):
        """Perform one optimization step for a replica.
        
        Args:
            replica (Dict): Replica state dictionary.
        """
        # Perturb data (already includes boundary reflection)
        perturbed = perturb_vectors(
            replica['current_data'],
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
            objective, replica['current_objective'], replica['temperature']
        )
        
        if accept:
            # Update current state
            replica['current_data'] = perturbed.copy()
            replica['current_objective'] = objective
            
            # Update best if better
            if self._is_better(objective, replica['best_objective']):
                replica['best_data'] = perturbed.copy()
                replica['best_objective'] = objective
            
            # Record step
            replica['metrics_history'].append((replica['step'], metrics, objective, replica['best_data'].copy()))
            replica['step'] += 1
        # Note: We only record accepted steps to avoid misleading history
        # where rejected steps would show the old objective with a new step number
        
        # Cool temperature
        replica['temperature'] *= (1 - self.cooling_rate)
    

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
                replica_i['current_objective'],
                replica_j['current_objective'],
                replica_i['temperature'],
                replica_j['temperature'],
                self.mode
            )
            
            if accepted:

                # Swap temperatures (not data!) to keep each replica's history continuous
                old_temp_i = replica_i['temperature']
                old_temp_j = replica_j['temperature']
                
                # Record temperature changes with current step number
                current_step = replica_i['step']  # Use step from replica i
                record_temperature_change(replica_i, old_temp_j, current_step)
                record_temperature_change(replica_j, old_temp_i, replica_j['step'])
                
                # Collect for database logging
                if self.db_enabled:
                    db_exchanges.append((current_step, i, old_temp_j))
                    db_exchanges.append((replica_j['step'], j, old_temp_i))
            
            # Record statistics
            record_exchange(replica_i, j, accepted)
            record_exchange(replica_j, i, accepted)
        
        # Log exchanges to database
        if self.db_enabled and db_exchanges:
            self.db_writer.insert_temperature_exchanges(db_exchanges)
    

    def _get_best_replica(self) -> Dict:
        """Find replica with best objective value."""
        if self.mode == 'maximize':
            return max(self.replicas, key=lambda r: r['best_objective'])
        elif self.mode == 'minimize':
            return min(self.replicas, key=lambda r: r['best_objective'])
        else:  # target mode
            return min(self.replicas, 
                      key=lambda r: abs(r['best_objective'] - self.target_value))
    

    def save_checkpoint(self, filepath: str):
        """Save current state to checkpoint file."""
        checkpoint = {
            'replicas': [self._serialize_state(r) for r in self.replicas],
            'temperature_ladder': self.temperature_ladder.temperatures.tolist(),
            'elapsed_time': time.time() - self.replicas[0]['start_time'],
            'hyperparameters': self.replicas[0]['hyperparameters'],
            'is_dataframe': self.is_dataframe,
            'column_names': self.column_names,
            'bounds': self.bounds,
            'exchange_interval': self.exchange_interval,
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
    def _serialize_state(state: Dict) -> Dict[str, Any]:
        """Return state dictionary for pickling (already in dict format)."""
        return state
    

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
            verbose=checkpoint.get('verbose', False),
            n_workers=checkpoint.get('n_workers')
        )
        
        # Restore states (already in dict format)
        climber.replicas = checkpoint['replicas']
        
        # Restore temperature ladder
        climber.temperature_ladder = TemperatureLadder(
            temperatures=np.array(checkpoint['temperature_ladder'])
        )
        
        # Get elapsed time from checkpoint
        elapsed_seconds = checkpoint['elapsed_time']
        
        # Reset temperatures if requested
        if reset_temperatures:
            for i, replica in enumerate(climber.replicas):
                replica['temperature'] = climber.temperature_ladder.temperatures[i]

            if climber.verbose:
                print(f"Resumed from checkpoint with reset temperatures: {elapsed_seconds/60:.1f} minutes elapsed")
        
        else:
            if climber.verbose:
                print(f"Resumed from checkpoint: {elapsed_seconds/60:.1f} minutes elapsed")
        
        # Restore bounds
        climber.bounds = checkpoint.get('bounds', (np.min(climber.data, axis=0), np.max(climber.data, axis=0)))
        
        # Adjust replica start times to account for elapsed time
        # When resuming, we want elapsed time calculations to continue from where they left off
        # So we set start_time = current_time - elapsed_time
        current_time = time.time()
        adjusted_start_time = current_time - elapsed_seconds
        
        for replica in climber.replicas:
            replica['start_time'] = adjusted_start_time
        
        return climber
