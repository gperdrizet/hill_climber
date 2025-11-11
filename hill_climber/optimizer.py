"""Hill climbing optimization with simulated annealing."""

import numpy as np
import pandas as pd
import pickle
import time
import os
from multiprocessing import Pool, cpu_count

from .climber_functions import perturb_vectors, calculate_objective
from .plotting_functions import plot_input_data, plot_results as plot_results_func


class HillClimber:
    """Hill climbing optimizer with optional simulated annealing.
    
    Performs optimization using hill climbing with optional simulated annealing
    for escaping local optima. Supports three optimization modes: maximize,
    minimize, or target a specific value.
    
    Supports n-dimensional data where objective function receives each column
    as a separate argument.
    
    Attributes:
        data: Initial data (numpy array or pandas DataFrame)
        objective_func: Objective function returning (metrics_dict, objective_value)
        max_time: Maximum runtime in minutes
        step_size: Maximum perturbation amount for each step
        perturb_fraction: Fraction of points to perturb at each step
        temperature: Initial temperature for simulated annealing (0 disables)
        cooling_rate: Multiplicative temperature decrease rate
        mode: Optimization mode ('maximize', 'minimize', or 'target')
        target_value: Target objective value when mode='target'
    """
    
    def __init__(
        self,
        data,
        objective_func,
        max_time=3,
        step_size=0.1,
        perturb_fraction=0.1,
        temperature=0.0,
        cooling_rate=0.995,
        mode='maximize',
        target_value=None,
        checkpoint_file=None,
        save_interval=60
    ):
        """Initialize HillClimber.
        
        Args:
            data: numpy array (N x M) or pandas DataFrame with M columns
            objective_func: Function that takes M column arrays and returns 
                          (metrics_dict, objective_value). For 2D data, receives (x, y).
                          For 3D data, receives (x, y, z), etc.
            max_time: Maximum runtime in minutes (default: 3)
            step_size: Maximum perturbation amount (default: 0.1)
            perturb_fraction: Fraction of points to perturb each step (default: 0.1)
            temperature: Initial temperature for simulated annealing (default: 0.0)
            cooling_rate: Temperature decrease rate (default: 0.995)
            mode: 'maximize', 'minimize', or 'target' (default: 'maximize')
            target_value: Target objective value for target mode (default: None)
            checkpoint_file: Path to save/load checkpoints (default: None)
            save_interval: Seconds between checkpoint saves (default: 60)
            
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
        self.step_size = step_size
        self.perturb_fraction = perturb_fraction
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.mode = mode
        self.target_value = target_value
        self.checkpoint_file = checkpoint_file
        self.save_interval = save_interval
        
        # These will be set during climb
        self.best_data = None
        self.current_data = None
        self.best_objective = None
        self.current_objective = None
        self.best_distance = None
        self.steps = None
        self.metrics = None
        self.step = 0
        self.temp = temperature
        self.start_time = None
        self.last_save_time = None


    def save_checkpoint(self, force=False):
        """Save current optimization state to checkpoint file.
        
        Args:
            force: Save even if save_interval hasn't elapsed (default: False)
        """
        if not self.checkpoint_file:
            return
            
        current_time = time.time()
        
        if not force and self.last_save_time is not None:
            if current_time - self.last_save_time < self.save_interval:
                return
        
        checkpoint_data = {
            'best_data': self.best_data.copy() if self.best_data is not None else None,
            'current_data': self.current_data.copy() if self.current_data is not None else None,
            'best_objective': self.best_objective,
            'current_objective': self.current_objective,
            'best_distance': self.best_distance,
            'steps': self.steps.copy() if self.steps is not None else None,
            'step': self.step,
            'temp': self.temp,
            'start_time': self.start_time,
            'elapsed_time': current_time - self.start_time if self.start_time else 0,
            'hyperparameters': {
                'max_time': self.max_time,
                'step_size': self.step_size,
                'perturb_fraction': self.perturb_fraction,
                'temperature': self.temperature,
                'cooling_rate': self.cooling_rate,
                'mode': self.mode,
                'target_value': self.target_value
            },
            'data_info': {
                'is_dataframe': self.is_dataframe,
                'columns': self.columns,
                'bounds': self.bounds
            },
            'original_data': self.data_numpy.copy()
        }
        
        # Create checkpoint directory if needed
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.last_save_time = current_time
        print(f"Checkpoint saved: {self.checkpoint_file}")


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
            
            # Restore optimization state
            self.best_data = checkpoint_data['best_data']
            self.current_data = checkpoint_data['current_data']
            self.best_objective = checkpoint_data['best_objective']
            self.current_objective = checkpoint_data['current_objective']
            self.best_distance = checkpoint_data['best_distance']
            self.steps = checkpoint_data['steps']
            self.step = checkpoint_data['step']
            self.temp = checkpoint_data['temp']
            self.start_time = checkpoint_data['start_time']
            
            # Restore data info
            data_info = checkpoint_data['data_info']
            self.is_dataframe = data_info['is_dataframe']
            self.columns = data_info['columns']
            self.bounds = data_info['bounds']
            self.data_numpy = checkpoint_data['original_data']
            
            # Adjust start time to account for elapsed time
            elapsed_time = checkpoint_data['elapsed_time']
            self.start_time = time.time() - elapsed_time
            
            print(f"Checkpoint loaded: {checkpoint_file}")
            print(f"Resuming from step {self.step}, elapsed time: {elapsed_time:.1f}s")
            
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
            step_size=hyperparams['step_size'],
            perturb_fraction=hyperparams['perturb_fraction'],
            temperature=hyperparams['temperature'],
            cooling_rate=hyperparams['cooling_rate'],
            mode=hyperparams['mode'],
            target_value=hyperparams['target_value'],
            checkpoint_file=new_checkpoint_file if new_checkpoint_file is not None else checkpoint_file
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

        # Initialize tracking structures if not resuming
        if self.steps is None:
            self.steps = {'Step': [], 'Objective value': [], 'Best_data': []}
            self.best_data = self.current_data = self.data_numpy.copy()
            
            # Get initial objective and dynamically create metric columns
            self.metrics, self.best_objective = calculate_objective(
                self.data_numpy, self.objective_func
            )

            for metric_name in self.metrics.keys():
                self.steps[metric_name] = []
            
            self.current_objective = self.best_objective

            self.best_distance = (
                abs(self.best_objective - self.target_value) 
                if self.mode == 'target' else None
            )
            
            self.step = 0
            self.temp = self.temperature
        
        # Set or update start time
        if self.start_time is None:
            self.start_time = time.time()
        
        # Main optimization loop
        while time.time() - self.start_time < self.max_time * 60:

            self.step += 1
            
            # Generate and evaluate new candidate solution
            new_data = perturb_vectors(
                self.current_data, 
                self.step_size, 
                self.perturb_fraction,
                self.bounds
            )

            self.metrics, new_objective = calculate_objective(new_data, self.objective_func)
            
            # Determine if we accept the new solution
            if self.temperature > 0:

                # Simulated annealing: accept worse solutions probabilistically
                delta = self._calculate_delta(new_objective)
                accept = delta >= 0 or np.random.random() < np.exp(delta / max(self.temp, 1e-10))
                
                if accept:

                    self.current_data = new_data
                    self.current_objective = new_objective
                    
                    # Update best if this is an improvement
                    if self._is_improvement(new_objective):

                        self.best_data = new_data.copy()
                        self.best_objective = new_objective

                        if self.mode == 'target':
                            self.best_distance = abs(new_objective - self.target_value)

                        self._record_improvement()
                
                self.temp *= self.cooling_rate

            else:
                # Standard hill climbing: only accept improvements
                if self._is_improvement(new_objective):
                    self.best_data = self.current_data = new_data
                    self.best_objective = self.current_objective = new_objective

                    if self.mode == 'target':
                        self.best_distance = abs(new_objective - self.target_value)

                    self._record_improvement()
            
            # Save checkpoint periodically
            self.save_checkpoint()
        
        # Save final checkpoint
        self.save_checkpoint(force=True)
        
        # Convert back to DataFrame if input was DataFrame
        best_data_output = (
            pd.DataFrame(self.best_data, columns=self.columns) 
            if self.is_dataframe else self.best_data
        )
        
        return best_data_output, pd.DataFrame(self.steps)


    def climb_parallel(self, replicates=4, initial_noise=0.0, output_file=None, 
                      checkpoint_dir=None):
        """Run hill climbing in parallel with multiple replicates.
        
        Args:
            replicates: Number of parallel replicates to run (default: 4)
            initial_noise: Std dev of Gaussian noise added to initial data (default: 0.0)
            output_file: Path to save results as pickle file (default: None)
            checkpoint_dir: Directory to save individual replicate checkpoints (default: None)
            
        Returns:
            List of (best_data, steps_df) tuples, one for each replicate
            
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

                # Add uniform noise and clip to ensure non-negative values
                noise = np.random.uniform(-initial_noise, initial_noise, new_data.shape)
                new_data = np.maximum(0.0, new_data + noise)

            replicate_inputs.append(new_data)
        
        # Package arguments for parallel execution
        args_list = []
        for i, data_rep in enumerate(replicate_inputs):
            checkpoint_file = None
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, f'replicate_{i:03d}.pkl')
            
            args_list.append((
                data_rep, self.objective_func, self.max_time, self.step_size,
                self.perturb_fraction, self.temperature, self.cooling_rate,
                self.mode, self.target_value, self.is_dataframe, self.columns,
                checkpoint_file, self.save_interval
            ))
        
        # Execute in parallel
        with Pool(processes=replicates) as pool:
            results = pool.map(_climb_wrapper, args_list)
        
        # Save results if requested
        if output_file:

            results_package = {
                'results': results,
                'hyperparameters': {
                    'max_time': self.max_time,
                    'step_size': self.step_size,
                    'perturb_fraction': self.perturb_fraction,
                    'replicates': replicates,
                    'initial_noise': initial_noise,
                    'temperature': self.temperature,
                    'cooling_rate': self.cooling_rate,
                    'objective_function': self.objective_func.__name__,
                    'mode': self.mode,
                    'target_value': self.target_value,
                    'input_size': len(self.data)
                },
                'input_data': self.data
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
            return new_obj > self.best_objective

        elif self.mode == 'minimize':
            return new_obj < self.best_objective

        else:  # target mode
            return abs(new_obj - self.target_value) < self.best_distance

    def _calculate_delta(self, new_obj):
        """Calculate delta for simulated annealing acceptance.
        
        Args:
            new_obj: New objective value
            
        Returns:
            Delta (positive = improvement, negative = deterioration)
        """

        if self.mode == 'maximize':
            return new_obj - self.current_objective

        elif self.mode == 'minimize':
            return self.current_objective - new_obj

        else:  # target mode
            curr_dist = abs(self.current_objective - self.target_value)
            new_dist = abs(new_obj - self.target_value)

            return curr_dist - new_dist

    def _record_improvement(self):
        """Record current best solution in steps history."""

        self.steps['Step'].append(self.step)
        self.steps['Objective value'].append(self.best_objective)
        self.steps['Best_data'].append(self.best_data.copy())
        
        for metric_name, metric_value in self.metrics.items():
            self.steps[metric_name].append(metric_value)


def _climb_wrapper(args):
    """Wrapper for parallel execution of HillClimber.climb().
    
    Args:
        args: Tuple of (data_numpy, objective_func, max_time, step_size, 
              perturb_fraction, temperature, cooling_rate, mode, target_value, 
              is_dataframe, columns, checkpoint_file, save_interval)
        
    Returns:
        Result from climb(): (best_data, steps_df)
    """

    (data_numpy, objective_func, max_time, step_size, perturb_fraction, 
     temperature, cooling_rate, mode, target_value, is_dataframe, columns,
     checkpoint_file, save_interval) = args
    
    # Reconstruct original data format for HillClimber
    data_input = (
        pd.DataFrame(data_numpy, columns=columns) 
        if is_dataframe else data_numpy
    )
    
    climber = HillClimber(
        data=data_input,
        objective_func=objective_func,
        max_time=max_time,
        step_size=step_size,
        perturb_fraction=perturb_fraction,
        temperature=temperature,
        cooling_rate=cooling_rate,
        mode=mode,
        target_value=target_value,
        checkpoint_file=checkpoint_file,
        save_interval=save_interval
    )
    
    return climber.climb()
