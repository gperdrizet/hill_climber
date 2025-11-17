"""Hill climbing optimization with simulated annealing."""

import numpy as np
import pandas as pd
import pickle
import time
import os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from .climber_functions import perturb_vectors, calculate_objective
from .plotting_functions import plot_input_data, plot_results as plot_results_func
from .optimizer_state import OptimizerState


class HillClimber:
    """Hill climbing optimizer with optional simulated annealing.
    
    Performs optimization using hill climbing with optional simulated annealing
    for escaping local optima. Supports three optimization modes: maximize,
    minimize, or target a specific value.
    
    Works with multi-column datasets where the objective function receives each
    column as a separate argument. Uses a unified OptimizerState dataclass to
    manage all optimization progress tracking internally.
    
    Attributes:
        data: Initial data (numpy array or pandas DataFrame)
        objective_func: Objective function returning (metrics_dict, objective_value)
        max_time: Maximum runtime in minutes
        perturb_fraction: Fraction of points to perturb at each step
        step_spread: Standard deviation of normal distribution for perturbations
        temperature: Initial temperature for simulated annealing (0 disables)
        cooling_rate: Multiplicative cooling factor (computed from input parameter)
        mode: Optimization mode ('maximize', 'minimize', or 'target')
        target_value: Target objective value when mode='target'
        state: OptimizerState instance managing all optimization progress
    """
    
    def __init__(
        self,
        data,
        objective_func,
        max_time=30,
        perturb_fraction=0.05,
        temperature=1000,
        cooling_rate=0.000001,
        mode='maximize',
        target_value=None,
        checkpoint_file=None,
        save_interval=60,
        plot_progress=None,
        step_spread=1.0
    ):
        """Initialize HillClimber.
        
        Args:
            data: numpy array ``(N x M)`` or pandas DataFrame with M columns
            objective_func: Function that takes M column arrays and returns 
                          ``(metrics_dict, objective_value)``. For 2-column data, receives ``(x, y)``.
                          For 3-column data, receives ``(x, y, z)``, etc.
            max_time: Maximum runtime in minutes (default: 30)
            perturb_fraction: Fraction of points to perturb each step (default: 0.05)
            temperature: Initial temperature for simulated annealing (default: 1000)
            cooling_rate: Amount subtracted from 1 to get multiplicative cooling rate.
                         For example, ``0.000001`` results in ``temp *= 0.999999`` each step.
                         Smaller values = slower cooling. (default: 0.000001)
            mode: ``'maximize'``, ``'minimize'``, or ``'target'`` (default: ``'maximize'``)
            target_value: Target objective value for target mode (default: None)
            checkpoint_file: Path to save/load checkpoints (default: None)
            save_interval: Seconds between checkpoint saves (default: 60)
            plot_progress: Plot results every N minutes during optimization. 
                          If None (default), no plots are drawn during optimization.
            step_spread: Standard deviation of normal distribution for perturbations (default: 1.0)
            
        Raises:
            ValueError: If mode is invalid or target_value missing for target mode
        """

        if mode not in ['maximize', 'minimize', 'target']:
            raise ValueError(f"Mode must be 'maximize', 'minimize', or 'target', got '{mode}'")
        
        if mode == 'target' and target_value is None:
            raise ValueError("target_value must be specified when mode='target'")
        
        # Store original data and format info
        self.data = data
        self.is_dataframe = isinstance(data, pd.DataFrame)
        self.columns = data.columns.tolist() if self.is_dataframe else None
        
        # Convert to numpy for efficient processing during optimization
        self.data_numpy = data.values if self.is_dataframe else data.copy()
        
        # Calculate bounds for perturbation clipping
        self.min_bounds = np.min(self.data_numpy, axis=0)
        self.max_bounds = np.max(self.data_numpy, axis=0)
        self.bounds = (self.min_bounds, self.max_bounds)
        
        self.objective_func = objective_func
        self.max_time = max_time
        self.perturb_fraction = perturb_fraction
        self.temperature = temperature

        # Convert user-provided cooling_rate to multiplicative factor
        # User specifies 1 - multiplicative_rate, we store the multiplicative rate
        self.cooling_rate = 1 - cooling_rate
        self.cooling_rate_input = cooling_rate  # Store original for checkpointing
        self.mode = mode
        self.target_value = target_value
        self.checkpoint_file = checkpoint_file
        self.save_interval = save_interval
        self.plot_progress = plot_progress
        self.step_spread = step_spread
        
        # Unified state management via dataclass
        self.state = OptimizerState()


    def save_checkpoint(self, force=False):
        """Save current optimization state to checkpoint file.
        
        Args:
            force: Save even if save_interval hasn't elapsed (default: False)
        """
        if not self.checkpoint_file:
            return
            
        current_time = time.time()
        
        if not force and self.state.last_save_time is not None:
            if current_time - self.state.last_save_time < self.save_interval:
                return
        
        # Get complete state data from dataclass (includes hyperparameters and original_data)
        state_dict = self.state.to_checkpoint_dict()
        
        checkpoint_data = {
            'state': state_dict,
            'elapsed_time': current_time - self.state.start_time if self.state.start_time else 0,
            'data_info': {
                'is_dataframe': self.is_dataframe,
                'columns': self.columns,
                'bounds': self.bounds
            }
        }
        
        # Create checkpoint directory if needed
        checkpoint_dir = os.path.dirname(self.checkpoint_file)

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.state.last_save_time = current_time
        print(f"Checkpoint saved: {self.checkpoint_file}")


    def plot_progress_check(self, force=False):
        """Plot optimization progress if plot_progress interval has elapsed.
        
        Args:
            force: Plot even if plot_progress interval hasn't elapsed (default: False)
        """

        if self.plot_progress is None:
            return
        
        if self.state.start_time is None:
            return
            
        current_time = time.time()
        
        if not force and self.state.last_plot_time is not None:
            if (current_time - self.state.last_plot_time) / 60 < self.plot_progress:
                return
        
        # Clear any existing plots
        plt.close('all')
        
        # Clear output in Jupyter notebooks to replace previous plot
        try:
            from IPython.display import clear_output
            clear_output(wait=True)

        except ImportError:
            # Not in IPython/Jupyter environment
            pass
        
        # Get results from state
        best_data_output, history_df = self.state.get_results(
            is_dataframe=self.is_dataframe,
            columns=self.columns
        )
        
        # Format as expected by plot_results (single replicate)
        results = {
            'input_data': self.data,
            'results': [(self.data, best_data_output, history_df)]
        }
        
        # Plot current progress
        elapsed_min = (current_time - self.state.start_time) / 60
        last_elapsed_min = (self.state.last_plot_time - self.state.start_time) / 60 if self.state.last_plot_time else 0
        
        # Format elapsed time based on duration
        def format_elapsed(minutes):

            if minutes < 60:
                return f"{int(minutes)} minutes"

            else:
                hours = minutes / 60
                return f"{hours:.1f} hours"
        
        # Check if there are any steps to plot
        if not self.state.has_steps():
            print(f"\nNo accepted steps since last progress update")
            print(f"Last progress update: {format_elapsed(last_elapsed_min)}")
            print(f"Current time: {format_elapsed(elapsed_min)}")

        else:
            print(f"\nPlotting progress at {format_elapsed(elapsed_min)}...")
            plot_results_func(results, plot_type='scatter')
        
        self.state.last_plot_time = current_time


    def load_checkpoint(self, checkpoint_file):
        """Load optimization state from checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file to load
            
        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """

        if not os.path.exists(checkpoint_file):
            print(f"Checkpoint file not found: {checkpoint_file}")
            return False
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Check if this is a new-format checkpoint with 'state' key
            if 'state' in checkpoint_data:
                # New format: load from dataclass
                self.state = OptimizerState.from_checkpoint_dict(checkpoint_data['state'])
                
                # Restore data info (needed for HillClimber operations)
                data_info = checkpoint_data.get('data_info', {})
                self.is_dataframe = data_info.get('is_dataframe', False)
                self.columns = data_info.get('columns')
                self.bounds = data_info.get('bounds')
                
                # Restore original data from state
                if self.state.original_data is not None:
                    self.data = self.state.original_data
                    self.data_numpy = self.data.values if self.is_dataframe else self.data
                else:
                    # This should not happen with properly saved checkpoints
                    raise ValueError("Checkpoint missing original_data in state")
                        
            else:
                # Old format: migrate to dataclass
                self.state = OptimizerState()
                self.state.best_data = checkpoint_data['best_data']
                self.state.current_data = checkpoint_data['current_data']
                self.state.best_objective = checkpoint_data['best_objective']
                self.state.current_objective = checkpoint_data['current_objective']
                self.state.best_distance = checkpoint_data['best_distance']
                self.state.history = checkpoint_data['steps']
                self.state.step = checkpoint_data['step']
                self.state.temperature = checkpoint_data['temp']
                self.state.start_time = checkpoint_data['start_time']
                
                # Store hyperparameters and original data in state for consistency
                self.state.hyperparameters = checkpoint_data.get('hyperparameters', {})
                self.state.original_data = checkpoint_data.get('original_data')
                
                # Restore data info
                data_info = checkpoint_data['data_info']
                self.is_dataframe = data_info['is_dataframe']
                self.columns = data_info['columns']
                self.bounds = data_info['bounds']
                self.data_numpy = checkpoint_data['original_data']
                self.data = self.state.original_data
            
            # Adjust start time to account for elapsed time
            elapsed_time = checkpoint_data['elapsed_time']
            self.state.start_time = time.time() - elapsed_time
            
            print(f"Checkpoint loaded: {checkpoint_file}")
            print(f"Resuming from step {self.state.step}, elapsed time: {elapsed_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False


    @classmethod
    def resume_from_checkpoint(cls, checkpoint_file, objective_func, 
                             new_max_time=None, new_checkpoint_file=None):
        """Create a new HillClimber instance from a checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            objective_func: Objective function (must be same as original)
            new_max_time: New max_time for continued optimization (default: use original)
            new_checkpoint_file: New checkpoint file path (default: use original)
            
        Returns:
            New HillClimber instance with state loaded from checkpoint
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Extract hyperparameters
        hyperparams = checkpoint_data['hyperparameters']
        data_info = checkpoint_data['data_info']
        
        # Reconstruct original data
        original_data = checkpoint_data['original_data']

        if data_info['is_dataframe']:
            original_data = pd.DataFrame(original_data, columns=data_info['columns'])
        
        # Create new instance
        climber = cls(
            data=original_data,
            objective_func=objective_func,
            max_time=new_max_time if new_max_time is not None else hyperparams['max_time'],
            perturb_fraction=hyperparams['perturb_fraction'],
            temperature=hyperparams['temperature'],
            cooling_rate=hyperparams['cooling_rate'],
            mode=hyperparams['mode'],
            target_value=hyperparams['target_value'],
            checkpoint_file=new_checkpoint_file if new_checkpoint_file is not None else checkpoint_file,
            step_spread=hyperparams['step_spread']
        )
        
        # Load the checkpoint state
        climber.load_checkpoint(checkpoint_file)
        
        return climber


    def climb(self):
        """Perform hill climbing optimization.
        
        Returns:
            Tuple of (best_data, steps_df) where:
                - best_data: Best solution found (same format as input data)
                - steps_df: DataFrame tracking optimization progress
        """

        # Initialize state if not resuming
        if self.state.current_data is None:
            # Get initial objective and metrics
            metrics, objective = calculate_objective(
                self.data_numpy, self.objective_func
            )
            
            # Prepare hyperparameters dictionary
            hyperparameters = {
                'max_time': self.max_time,
                'perturb_fraction': self.perturb_fraction,
                'temperature': self.temperature,
                'cooling_rate': self.cooling_rate_input,
                'mode': self.mode,
                'target_value': self.target_value,
                'step_spread': self.step_spread,
                'checkpoint_file': self.checkpoint_file,
                'save_interval': self.save_interval,
                'plot_progress': self.plot_progress
            }
            
            # Initialize state
            self.state.initialize(
                data=self.data_numpy,
                objective=objective,
                metrics=metrics,
                temperature=self.temperature,
                target_value=self.target_value,
                mode=self.mode,
                original_data=self.data,
                hyperparameters=hyperparameters
            )
        
        # Set or update start time
        if self.state.start_time is None:
            self.state.start_time = time.time()
        
        # Main optimization loop
        while time.time() - self.state.start_time < self.max_time * 60:

            self.state.step += 1
            
            # Generate and evaluate new candidate solution
            new_data = perturb_vectors(
                self.state.current_data, 
                self.perturb_fraction,
                self.bounds,
                self.step_spread
            )

            metrics, new_objective = calculate_objective(new_data, self.objective_func)
            
            # Determine if we accept the new solution
            if self.temperature > 0:

                # Simulated annealing: accept worse solutions probabilistically
                delta = self._calculate_delta(new_objective)
                accept = delta >= 0 or np.random.random() < np.exp(delta / max(self.state.temperature, 1e-10))
                
                if accept:
                    self.state.update_current(new_data, new_objective, metrics)
                    
                    # Update best if this is an improvement
                    if self._is_improvement(new_objective):
                        self.state.update_best(new_data, new_objective, self.target_value)
                        self.state.record_improvement()
                
                self.state.temperature *= self.cooling_rate

            else:
                # Standard hill climbing: only accept improvements
                if self._is_improvement(new_objective):
                    # Update both current and best
                    self.state.update_current(new_data, new_objective, metrics)
                    self.state.update_best(new_data, new_objective, self.target_value)
                    self.state.record_improvement()
            
            # Save checkpoint periodically
            self.save_checkpoint()
            
            # Plot progress periodically
            self.plot_progress_check()
        
        # Save final checkpoint
        self.save_checkpoint(force=True)
        
        # Plot final results
        self.plot_progress_check(force=True)
        
        # Return results
        return self.state.get_results(
            is_dataframe=self.is_dataframe,
            columns=self.columns
        )


    def climb_parallel(self, replicates=4, initial_noise=0.0, output_file=None, 
                      checkpoint_dir=None):
        """Run hill climbing in parallel with multiple replicates.
        
        Args:
            replicates: Number of parallel replicates to run (default: 4)
            initial_noise: Std dev of Gaussian noise added to initial data (default: 0.0)
            output_file: Path to save results as pickle file (default: None)
            checkpoint_dir: Directory to save individual replicate checkpoints (default: None)
            
        Returns:
            Dictionary with:
                'input_data': Original input data (before noise)
                'results': List of (noisy_initial_data, best_data, steps_df) tuples
            
        Raises:
            ValueError: If replicates exceeds available CPU count
        """

        # Validate CPU availability
        available_cpus = cpu_count()

        if replicates > available_cpus:
            raise ValueError(
                f"Replicates ({replicates}) exceeds CPU count ({available_cpus}). "
                f"Reduce replicates or use more CPUs."
            )
        
        # Create replicate inputs with optional noise
        replicate_inputs = []

        for _ in range(replicates):

            new_data = self.data_numpy.copy()

            if initial_noise > 0:

                # Add uniform noise and reflect values back into bounds
                noise = np.random.uniform(-initial_noise, initial_noise, new_data.shape)
                new_data = new_data + noise
                
                # Reflect values that exceed bounds back into valid range
                # This prevents accumulation at boundaries
                for col_idx in range(new_data.shape[1]):

                    col_min = self.min_bounds[col_idx]
                    col_max = self.max_bounds[col_idx]
                    col_range = col_max - col_min
                    
                    # Reflect values below minimum
                    below_min = new_data[:, col_idx] < col_min
                    new_data[below_min, col_idx] = col_min + (col_min - new_data[below_min, col_idx])
                    
                    # Reflect values above maximum
                    above_max = new_data[:, col_idx] > col_max
                    new_data[above_max, col_idx] = col_max - (new_data[above_max, col_idx] - col_max)
                    
                    # Handle cases where reflection itself goes out of bounds
                    # (can happen with large noise values) - wrap around
                    still_out = (new_data[:, col_idx] < col_min) | (new_data[:, col_idx] > col_max)

                    if np.any(still_out):

                        # Use modulo to wrap into range
                        new_data[still_out, col_idx] = col_min + np.mod(
                            new_data[still_out, col_idx] - col_min, col_range
                        )

            replicate_inputs.append(new_data)
        
        # Package arguments for parallel execution
        args_list = []

        for i, data_rep in enumerate(replicate_inputs):

            checkpoint_file = None

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, f'replicate_{i:03d}.pkl')
            
            args_list.append((
                data_rep, self.objective_func, self.max_time,
                self.perturb_fraction, self.temperature, self.cooling_rate,
                self.mode, self.target_value, self.is_dataframe, self.columns,
                checkpoint_file, self.save_interval, None,  # Disable plot_progress for parallel
                self.step_spread
            ))
        
        # Execute in parallel
        with Pool(processes=replicates) as pool:
            optimization_results = pool.map(_climb_wrapper, args_list)
        
        # Combine results with noisy initial data for each replicate
        # Format: [(noisy_initial_data, best_data, steps_df), ...]
        results_list = []
        for i, (best_data, steps_df) in enumerate(optimization_results):
            # Convert noisy initial numpy array to DataFrame if needed
            if self.is_dataframe:
                noisy_initial = pd.DataFrame(replicate_inputs[i], columns=self.columns)
            else:
                noisy_initial = replicate_inputs[i]
            
            results_list.append((noisy_initial, best_data, steps_df))
        
        # Package results with original input data (saved only once)
        results = {
            'input_data': self.data,  # Original data before noise
            'results': results_list    # List of (noisy_initial, best, steps) tuples
        }
        
        # Save results if requested
        if output_file:

            results_package = {
                'input_data': self.data,
                'results': results_list,
                'hyperparameters': {
                    'max_time': self.max_time,
                    'perturb_fraction': self.perturb_fraction,
                    'replicates': replicates,
                    'initial_noise': initial_noise,
                    'temperature': self.temperature,
                    'cooling_rate': self.cooling_rate_input,
                    'objective_function': self.objective_func.__name__,
                    'mode': self.mode,
                    'target_value': self.target_value,
                    'input_size': len(self.data),
                    'step_spread': self.step_spread
                }
            }
            
            with open(output_file, 'wb') as f:
                pickle.dump(results_package, f)

            print(f"Results saved to: {output_file}")
        
        return results


    def plot_input(self, plot_type='scatter'):
        """Plot the input data distribution.
        
        Args:
            plot_type: 'scatter' or 'kde' (default: 'scatter')
        """

        plot_input_data(self.data, plot_type=plot_type)

    def plot_results(self, results, plot_type='scatter', metrics=None):
        """Visualize hill climbing results.
        
        Args:
            results: List of (best_data, steps_df) tuples from climb_parallel()
            plot_type: 'scatter' or 'histogram' (default: 'scatter')
            metrics: List of metric names to display (default: None shows all)
        """

        plot_results_func(results, plot_type=plot_type, metrics=metrics)


    def _is_improvement(self, new_obj):
        """Check if new objective is an improvement.
        
        Args:
            new_obj: New objective value
            
        Returns:
            True if new_obj improves on current best
        """
        if self.mode == 'maximize':
            return new_obj > self.state.best_objective

        elif self.mode == 'minimize':
            return new_obj < self.state.best_objective

        else:  # target mode
            return abs(new_obj - self.target_value) < self.state.best_distance

    def _calculate_delta(self, new_obj):
        """Calculate delta for simulated annealing acceptance.
        
        Args:
            new_obj: New objective value
            
        Returns:
            Delta (positive = improvement, negative = deterioration)
        """

        if self.mode == 'maximize':
            return new_obj - self.state.current_objective

        elif self.mode == 'minimize':
            return self.state.current_objective - new_obj

        else:  # target mode
            curr_dist = abs(self.state.current_objective - self.target_value)
            new_dist = abs(new_obj - self.target_value)

            return curr_dist - new_dist


def _climb_wrapper(args):
    """Wrapper for parallel execution of HillClimber.climb().
    
    Args:
        args: Tuple of (data_numpy, objective_func, max_time, 
              perturb_fraction, temperature, cooling_rate, mode, target_value, 
              is_dataframe, columns, checkpoint_file, save_interval, plot_progress, 
              step_spread)
        
    Returns:
        Result from climb(): (best_data, steps_df)
    """

    (data_numpy, objective_func, max_time, perturb_fraction, 
     temperature, cooling_rate, mode, target_value, is_dataframe, columns,
     checkpoint_file, save_interval, plot_progress, step_spread) = args
    
    # Reconstruct original data format for HillClimber
    data_input = (
        pd.DataFrame(data_numpy, columns=columns) 
        if is_dataframe else data_numpy
    )
    
    climber = HillClimber(
        data=data_input,
        objective_func=objective_func,
        max_time=max_time,
        perturb_fraction=perturb_fraction,
        temperature=temperature,
        cooling_rate=cooling_rate,
        mode=mode,
        target_value=target_value,
        checkpoint_file=checkpoint_file,
        save_interval=save_interval,
        plot_progress=plot_progress,
        step_spread=step_spread
    )
    
    return climber.climb()
