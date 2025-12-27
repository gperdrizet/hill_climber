"""Hill climbing optimization with replica exchange."""

import os
import pickle
import time
from typing import Callable, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import Pool as PoolType

from .optimizer_state import create_replica_state, record_temperature_change, record_exchange
from .climber_functions import perturb_vectors
from .replica_exchange import (
    TemperatureLadder, ExchangeScheduler, should_exchange
)
from .replica_worker import run_replica_steps
from .config import (
    OptimizerConfig,
    DEFAULT_T_MIN,
    DEFAULT_COOLING_RATE,
    DEFAULT_INITIAL_STEP_SPREAD,
    DEFAULT_FINAL_STEP_SPREAD,
    DEFAULT_PERTURB_FRACTION,
    DEFAULT_N_REPLICAS,
    DEFAULT_EXCHANGE_INTERVAL,
    DEFAULT_TEMPERATURE_SCHEME,
    DEFAULT_EXCHANGE_STRATEGY,
    DEFAULT_MAX_TIME,
    DEFAULT_MODE,
    DEFAULT_CHECKPOINT_INTERVAL,
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
        mode: 'maximize', 'minimize', or 'target' (default: 'maximize')
        target_value: Target value (only used if mode='target')
        max_time: Maximum runtime in minutes (default: 10.0)
        initial_step_spread: Initial perturbation spread as fraction of input range (default: 0.25 = 25%).
            Step values are sampled from a gaussian distribution with mean 0 and standard deviation
            calculated per-feature as: feature_range * initial_step_spread. Each feature uses its own
            range for more appropriate perturbations across different scales.
        final_step_spread: Final perturbation spread as fraction of input range (default: None). If
            specified, step spread linearly decreases from initial_step_spread to final_step_spread
            over the course of max_time, enabling time-based cooling for more refined optimization
            near the end of the run.
        perturb_fraction: Fraction of data points to perturb each step (default: 0.001)
        n_replicas: Number of replicas for parallel tempering (default: 4), setting to 1 runs
            simulated annealing without replica exchange
        T_min: Base temperature (will be used as T_min for ladder) (default: 0.0001)
        T_max: Maximum temperature for hottest replica (default: 100 * T_min)
        cooling_rate: Temperature decay rate per successful step (default: 1e-10)
        temperature_scheme: 'geometric' or 'linear' temperature spacing (default: 'geometric')
        exchange_interval: Steps between exchange attempts (default: 100)
        exchange_strategy: 'even_odd', 'random', or 'all_neighbors' (default: 'even_odd')
        checkpoint_file: Path to save checkpoints (default: None, no checkpointing)
        checkpoint_interval: Batches between checkpoint saves (default: 1, i.e., every batch)
        db_enabled: Enable database logging for dashboard (default: True)
        db_path: Path to SQLite database file (default: '../data/hill_climb.db')
        db_step_interval: Collect metrics every Nth step. Uses tiered sampling: 1 for exchange_interval<10,
            10 for 10-99, 100 for 100-999, 1000 for >=1000 (default: None, auto-calculated)
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
        initial_step_spread: float = DEFAULT_INITIAL_STEP_SPREAD,
        final_step_spread: Optional[float] = DEFAULT_FINAL_STEP_SPREAD,
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
        db_enabled: bool = True,
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
            initial_step_spread=initial_step_spread,
            final_step_spread=final_step_spread,
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
        self.initial_step_spread = config.initial_step_spread
        self.final_step_spread = config.final_step_spread
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
        
        # step_spread_absolute will be calculated in climb() using current bounds
        # This allows users to modify bounds after initialization
        self.step_spread_absolute = None
        
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
        
        # Print configuration summary
        self._print_settings()


    def _print_settings(self):
        """Print optimizer configuration settings."""
        print("=" * 70)
        print("HillClimber Configuration")
        print("=" * 70)
        print()
        print(f"Data shape:           {self.data.shape}")
        print(f"Optimization mode:    {self.mode}" + (f" (target={self.target_value})" if self.mode == 'target' else ""))
        print(f"Max runtime:          {self.max_time} minutes")
        print()
        print("Temperature settings:")
        print(f"  T_min:              {self.T_min}")
        print(f"  T_max:              {self.T_max}")
        print(f"  Cooling rate:       {self.cooling_rate}")
        print(f"  Temperature scheme: {self.temperature_scheme}")
        print()
        print("Replica exchange:")
        print(f"  Number of replicas: {self.n_replicas}")
        print(f"  Exchange interval:  {self.exchange_interval} steps")
        print(f"  Exchange strategy:  {self.exchange_strategy}")
        print(f"  Worker processes:   {self.n_workers}")
        print()
        print("Perturbation settings:")
        print(f"  Initial step spread: {self.initial_step_spread} (fraction of range)")

        if self.final_step_spread is not None:
            print(f"  Final step spread:   {self.final_step_spread} (fraction at end of run)")

        print(f"  Perturb fraction:    {self.perturb_fraction}")
        print()
        print("Database settings:")
        print(f"  Enabled:            {self.db_enabled}")

        if self.db_enabled:
            print(f"  Path:               {self.db_path}")
            print(f"  Step interval:      {self.db_step_interval}")

        print()

        if self.checkpoint_file:
            print(f"Checkpointing:        {self.checkpoint_file} (every {self.checkpoint_interval} batch)")

        print("=" * 70)


    def climb(self) -> np.ndarray:
        """Run replica exchange optimization.
        
        Returns:
            np.ndarray: Best configuration found across all replicas.
                If database is enabled, use the dashboard to view optimization history.
        """

        # Calculate absolute step_spread from current bounds
        # Done here so users can modify bounds after initialization
        data_range = self.bounds[1] - self.bounds[0]
        self.step_spread_absolute = self.initial_step_spread * data_range

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
        
        # Store initial ladder temperatures for theoretical cooling tracking
        self.initial_ladder_temps = sorted([r['temperature'] for r in self.replicas], reverse=True)
        
        # Initialize temperature ladder history in database
        if self.db_enabled:
            self.db_writer.initialize_temperature_ladder_history(self.initial_ladder_temps, batch_num=0)

        # Initialize exchange scheduler and statistics
        scheduler = ExchangeScheduler(self.n_replicas, self.exchange_strategy)
        
        # Do the run
        return self._climb_parallel(scheduler)

    
    def _climb_parallel(self, scheduler: ExchangeScheduler) -> np.ndarray:
        """Run optimization with parallel workers.
        
        Args:
            scheduler (ExchangeScheduler): Scheduler for replica exchange.
            
        Returns:
            np.ndarray: Best configuration from best replica.
        """

        start_time = time.time()
        avg_batch_time = 0.0
        
        # Create worker pool
        with Pool(processes=self.n_workers) as pool:
            while (time.time() - start_time) < (self.max_time * 60 - avg_batch_time):
                batch_start = time.time()
                
                # Run batch of steps in parallel
                self._parallel_step_batch(pool, self.exchange_interval, start_time)
                
                # Attempt exchanges if we are optimizing multiple replicas
                if self.n_replicas > 1:
                    self._exchange_round(scheduler)
                
                # Increment batch counter
                self.batch_counter += 1
                
                # Update temperature ladder history by applying cooling to previous batch
                if self.db_enabled:
                    self.db_writer.update_temperature_ladder_history(
                        self.batch_counter, 
                        self.cooling_rate, 
                        self.exchange_interval
                    )
                    
                    # Record batch statistics (step spread and acceptance rates)
                    # Calculate acceptance rates for each replica from the preceding batch
                    acceptance_rates = []
                    for replica in self.replicas:
                        # Acceptance rate = num_accepted / perturbation_num
                        if replica['perturbation_num'] > 0:
                            rate = replica['num_accepted'] / replica['perturbation_num']
                            acceptance_rates.append(rate)
                    
                    if acceptance_rates:
                        mean_acceptance = float(np.mean(acceptance_rates))
                        min_acceptance = float(np.min(acceptance_rates))
                        max_acceptance = float(np.max(acceptance_rates))
                        
                        # Calculate current step spread (same logic as replica_worker)
                        data_range = self.bounds[1] - self.bounds[0]
                        if start_time is not None and self.final_step_spread is not None:
                            elapsed_time = time.time() - start_time
                            progress = min(elapsed_time / (self.max_time * 60.0), 1.0)
                            step_spread_initial = self.initial_step_spread * data_range
                            step_spread_final = self.final_step_spread * data_range
                            current_step_spread = step_spread_initial + (step_spread_final - step_spread_initial) * progress
                        else:
                            current_step_spread = self.step_spread_absolute
                        
                        # Convert back to fraction (0-1) by dividing by data_range, then take mean across dimensions
                        step_spread_fraction = current_step_spread / data_range
                        step_spread_scalar = float(np.mean(step_spread_fraction))
                        
                        self.db_writer.insert_batch_statistics(
                            self.batch_counter,
                            step_spread_scalar,
                            mean_acceptance,
                            min_acceptance,
                            max_acceptance
                        )
                
                # Checkpoint after every checkpoint_interval batches
                if self.checkpoint_file and (self.batch_counter % self.checkpoint_interval == 0):
                    self.save_checkpoint(self.checkpoint_file)
                
                # Update average batch time
                batch_time = time.time() - batch_start
                if self.batch_counter == 1:
                    avg_batch_time = batch_time
                else:
                    # Running average: new_avg = old_avg + (new_value - old_avg) / (n + 1)
                    avg_batch_time = avg_batch_time + (batch_time - avg_batch_time) / self.batch_counter
        
        return self._finalize_results()
    

    def _parallel_step_batch(self, pool: PoolType, n_steps: int, start_time: float):
        """Execute n_steps for all replicas in parallel.
        
        Args:
            pool (PoolType): Multiprocessing pool for parallel execution.
            n_steps (int): Number of optimization steps to execute per replica.
            start_time (float): Start time of the optimization run for time-based step cooling.
        """

        # Serialize current replica states
        state_dicts = [r for r in self.replicas]
        
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
            db_config=db_config,
            start_time=start_time
        )
        
        # Execute in parallel
        updated_states = pool.map(worker_func, state_dicts)
        
        # Collect database buffers from all replicas
        all_perturbations = []
        all_accepted = []
        all_step_metrics = []
        all_improvements = []
        all_improvement_metrics = []

        for i, state_dict in enumerate(updated_states):

            # Extract and collect database buffers if present
            if 'db_buffers' in state_dict:
                buffers = state_dict.pop('db_buffers')
                all_perturbations.extend(buffers.get('perturbations', []))
                all_accepted.extend(buffers.get('accepted', []))
                all_step_metrics.extend(buffers.get('step_metrics', []))
                all_improvements.extend(buffers.get('improvements', []))
                all_improvement_metrics.extend(buffers.get('improvement_metrics', []))
            
            # Preserve temperature_history before updating
            temp_history = self.replicas[i]['temperature_history']
            self.replicas[i] = state_dict
            self.replicas[i]['temperature_history'] = temp_history
        
        # Flush all collected database buffers to database (single source of truth)
        if self.db_enabled:
            self.db_writer.insert_perturbations_batch(all_perturbations)
            self.db_writer.insert_accepted_steps_batch(all_accepted)
            self.db_writer.insert_step_metrics_batch(all_step_metrics)
            self.db_writer.insert_improvements_batch(all_improvements)
            self.db_writer.insert_improvement_metrics_batch(all_improvement_metrics)
            
            # Update replica status snapshot
            for replica in self.replicas:
                self.db_writer.update_replica_status(
                    replica_id=replica['replica_id'],
                    current_perturbation_num=replica['perturbation_num'],
                    num_accepted=replica['num_accepted'],
                    num_improvements=replica['num_improvements'],
                    temperature=replica['temperature'],
                    best_objective=replica['best_objective'],
                    current_objective=replica['current_objective']
                )
    

    def _finalize_results(self) -> np.ndarray:
        """Complete optimization and return results.
        
        Writes final database snapshots and returns best solution found.
        
        Returns:
            np.ndarray: Best configuration from best replica.
        """

        # Write final state to database to ensure dashboard shows final values
        if self.db_enabled:
            import time
            timestamp = time.time()
            
            final_perturbations = []
            final_improvements = []
            final_improvement_metrics = []
            
            for replica in self.replicas:
                # Write final perturbation snapshot
                final_perturbations.append((
                    replica['replica_id'],
                    replica['perturbation_num'],
                    replica['current_objective'],
                    False,  # not newly accepted (just a snapshot)
                    False,  # not a new improvement (just a snapshot)
                    replica['temperature'],
                    timestamp
                ))
                
                # Write final best as improvement
                final_improvements.append((
                    replica['replica_id'],
                    replica['perturbation_num'],
                    replica['best_objective'],
                    replica['temperature'],
                    timestamp
                ))
                
                # Write final best metrics
                if 'best_metrics' in replica:
                    for metric_name, metric_value in replica['best_metrics'].items():
                        final_improvement_metrics.append((
                            replica['replica_id'],
                            replica['perturbation_num'],
                            metric_name,
                            metric_value
                        ))
            
            # Flush final snapshots to database
            self.db_writer.insert_perturbations_batch(final_perturbations)
            self.db_writer.insert_improvements_batch(final_improvements)
            self.db_writer.insert_improvement_metrics_batch(final_improvement_metrics)
            
            # Update final replica status
            for replica in self.replicas:
                self.db_writer.update_replica_status(
                    replica_id=replica['replica_id'],
                    current_perturbation_num=replica['perturbation_num'],
                    num_accepted=replica['num_accepted'],
                    num_improvements=replica['num_improvements'],
                    temperature=replica['temperature'],
                    best_objective=replica['best_objective'],
                    current_objective=replica['current_objective']
                )
            
            # Mark run as complete
            self.db_writer.set_run_end_time()

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
        
        return best_data_output
    

    def get_replicas(self) -> tuple:
        """Get best data from all replicas.
        
        Returns:
            tuple: Tuple of DataFrames (or numpy arrays if input was numpy), one for each replica,
                containing the best data found by that replica. Ordered by replica_id.
        """
        if not self.replicas:
            raise RuntimeError("No replicas available. Run climb() first.")
        
        replica_results = []
        for replica in sorted(self.replicas, key=lambda r: r['replica_id']):
            if self.is_dataframe:
                replica_df = pd.DataFrame(replica['best_data'], columns=self.column_names)
                replica_results.append(replica_df)
            else:
                replica_results.append(replica['best_data'])
        
        return tuple(replica_results)
    

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
            'initial_step_spread': self.initial_step_spread,
            'final_step_spread': self.final_step_spread,
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
            'initial_step_spread': self.initial_step_spread,
            'final_step_spread': self.final_step_spread,
            'step_spread_absolute_initial': self.step_spread_absolute.copy(),
            'step_spread_absolute_final': self.final_step_spread * (self.bounds[1] - self.bounds[0]) if self.final_step_spread is not None else None
        }
        
        # Evaluate initial objective
        from .climber_functions import calculate_objective
        metrics, objective = calculate_objective(
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

            # Initialize new state counters
            state['perturbation_num'] = 0
            state['num_accepted'] = 0
            state['num_improvements'] = 0
            
            # Initialize best_metrics (excluding 'Objective' entries)
            state['best_metrics'] = {k: v for k, v in metrics.items() if 'Objective' not in k}
            
            self.replicas.append(state)
    

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
                
                # Record temperature changes with current perturbation number
                current_pnum = replica_i['perturbation_num']  # Use perturbation_num from replica i
                record_temperature_change(replica_i, old_temp_j, current_pnum)
                record_temperature_change(replica_j, old_temp_i, replica_j['perturbation_num'])
                
                # Collect for database logging
                if self.db_enabled:
                    db_exchanges.append((current_pnum, i, old_temp_j))
                    db_exchanges.append((replica_j['perturbation_num'], j, old_temp_i))
            
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
            'replicas': [r for r in self.replicas],
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
            # Handle both old (step_spread) and new (initial_step_spread) names for backward compatibility
            initial_step_spread=hyperparams.get('initial_step_spread', hyperparams.get('step_spread', 0.25)),
            final_step_spread=hyperparams.get('final_step_spread'),
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
