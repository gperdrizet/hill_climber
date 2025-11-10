"""Hill climbing optimization with simulated annealing."""

import numpy as np
import pandas as pd
import pickle
import time
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
        target_value=None
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
        
        self.objective_func = objective_func
        self.max_time = max_time
        self.step_size = step_size
        self.perturb_fraction = perturb_fraction
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.mode = mode
        self.target_value = target_value
        
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


    def climb(self):
        """Perform hill climbing optimization.
        
        Returns:
            Tuple of (best_data, steps_df) where:
                - best_data: Best solution found (same format as input data)
                - steps_df: DataFrame tracking optimization progress
        """
        # Initialize tracking structures
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
        start_time = time.time()
        
        # Main optimization loop
        while time.time() - start_time < self.max_time * 60:
            self.step += 1
            
            # Generate and evaluate new candidate solution
            new_data = perturb_vectors(self.current_data, step_size=self.step_size, 
                                      perturb_fraction=self.perturb_fraction)
            self.metrics, new_objective = calculate_objective(
                new_data, self.objective_func
            )
            
            # Simulated annealing: accept worse solutions probabilistically
            if self.temperature > 0:
                delta = self._calculate_delta(new_objective)
                accept = delta >= 0 or np.random.random() < np.exp(delta / max(self.temp, 1e-10))
                
                if accept:
                    self.current_data = new_data
                    self.current_objective = new_objective
                    
                    # Update best if this accepted solution is the best so far
                    if self._is_improvement(self.current_objective):
                        self.best_data = self.current_data.copy()
                        self.best_objective = self.current_objective

                        if self.mode == 'target':
                            self.best_distance = abs(self.best_objective - self.target_value)

                        self._record_improvement()
                
                # Decrease temperature
                self.temp *= self.cooling_rate
            
            # Standard hill climbing: only accept improvements
            else:
                if self._is_improvement(new_objective):
                    self.best_data = new_data
                    self.best_objective = new_objective
                    self.current_data = new_data
                    self.current_objective = new_objective

                    if self.mode == 'target':
                        self.best_distance = abs(self.best_objective - self.target_value)

                    self._record_improvement()
        
        # Convert best_data back to DataFrame if input was DataFrame
        if self.is_dataframe:
            best_data_output = pd.DataFrame(self.best_data, columns=self.columns)
        else:
            best_data_output = self.best_data
        
        return best_data_output, pd.DataFrame(self.steps)


    def climb_parallel(
        self,
        replicates=4,
        initial_noise=0.0,
        output_file=None
    ):
        """Run hill climbing in parallel with multiple replicates.
        
        Executes multiple independent hill climbing runs in parallel, optionally
        adding noise to initial conditions for diversity. Results can be saved
        automatically to a pickle file.
        
        Args:
            replicates: Number of parallel replicates to run (default: 4)
            initial_noise: Std dev of Gaussian noise added to initial data (default: 0.0)
            output_file: Path to save results as pickle file (default: None, no save)
            
        Returns:
            List of (best_data, steps_df) tuples, one for each replicate
            
        Raises:
            ValueError: If replicates exceeds available CPU count
            
        Note:
            If output_file is provided, saves a dictionary containing:
                - 'results': List of (best_data, steps_df) tuples
                - 'hyperparameters': Dictionary of all run parameters
                - 'input_data': The original input data
        """

        # Validate CPU availability
        available_cpus = cpu_count()

        if replicates > available_cpus:

            raise ValueError(
                f"Replicates ({replicates}) exceeds CPU count ({available_cpus}). "
                f"Reduce replicates or use more CPUs."
            )
        
        # Create replicate inputs with optional noise for diversity (using numpy)
        replicate_inputs = []
        
        for _ in range(replicates):

            new_data = self.data_numpy.copy()

            if initial_noise > 0:
                noise = np.random.normal(0, initial_noise, new_data.shape)
                new_data = new_data + noise
    
            replicate_inputs.append(new_data)
        
        # Package arguments for parallel execution
        args_list = [
            (
                data_rep,
                self.objective_func,
                self.max_time,
                self.step_size,
                self.perturb_fraction,
                self.temperature,
                self.cooling_rate,
                self.mode,
                self.target_value,
                self.is_dataframe,
                self.columns
            )
            for data_rep in replicate_inputs
        ]
        
        # Execute replicates in parallel
        with Pool(processes=replicates) as pool:
            results = pool.map(_climb_wrapper, args_list)
        
        # Save results package if output file specified
        if output_file is not None:

            hyperparameters = {
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
            }
            
            results_package = {
                'results': results,
                'hyperparameters': hyperparameters,
                'input_data': self.data
            }
            
            with open(output_file, 'wb') as f:
                pickle.dump(results_package, f)
            
            print(f"Results saved to: {output_file}")
        
        return results
    
    def plot_input(self, plot_type='scatter'):
        """Plot the input data distribution.
        
        Displays a visualization of the input data showing the distribution
        of both variables.
        
        Args:
            plot_type: Type of plot - 'scatter' or 'kde' (default: 'scatter')
                       'scatter' shows x vs y scatter plot
                       'kde' shows Kernel Density Estimation plots
        """

        plot_input_data(self.data, plot_type=plot_type)


    def plot_results(self, results, plot_type='scatter', metrics=None):
        """Visualize hill climbing results.
        
        Creates a comprehensive visualization showing progress and snapshots
        for each replicate.
        
        Args:
            results: List of (best_data, steps_df) tuples from climb_parallel()
            plot_type: Type of snapshot plots - 'scatter' or 'histogram' (default: 'scatter')
                       Note: 'histogram' uses KDE (Kernel Density Estimation) plots
            metrics: List of metric names to display in progress plots and snapshots.
                     If None (default), all available metrics are shown.
                     Example: ['Pearson', 'Spearman'] or ['Mean X', 'Std X']
        """

        plot_results_func(results, plot_type=plot_type, metrics=metrics)


    def _is_improvement(self, new_obj):
        """Check if new objective represents an improvement.
        
        Args:
            new_obj: New objective value
            
        Returns:
            True if new_obj is an improvement over current best
        """

        if self.mode == 'maximize':
            return new_obj > self.best_objective

        elif self.mode == 'minimize':
            return new_obj < self.best_objective

        else:  # target mode
            new_dist = abs(new_obj - self.target_value)
            return new_dist < self.best_distance


    def _calculate_delta(self, new_obj):
        """Calculate delta for simulated annealing acceptance probability.
        
        Returns positive delta for improvements, negative for deteriorations.
        
        Args:
            new_obj: New objective value
            
        Returns:
            Delta value for acceptance probability calculation
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
        """Record current best solution and metrics in steps history."""

        self.steps['Step'].append(self.step)
        self.steps['Objective value'].append(self.best_objective)

        for metric_name, metric_value in self.metrics.items():
            self.steps[metric_name].append(metric_value)

        self.steps['Best_data'].append(self.best_data.copy())


def _climb_wrapper(args):
    """Wrapper for parallel execution of HillClimber.climb().
    
    Args:
        args: Tuple of (data_numpy, objective_func, max_time, step_size, perturb_fraction,
              temperature, cooling_rate, mode, target_value, is_dataframe, columns)
        
    Returns:
        Result from climb(): (best_data, steps_df)
    """

    (data_numpy, objective_func, max_time, step_size, perturb_fraction, 
     temperature, cooling_rate, mode, target_value, is_dataframe, columns) = args
    
    # Reconstruct original data format for HillClimber initialization
    if is_dataframe:
        data_input = pd.DataFrame(data_numpy, columns=columns)
    else:
        data_input = data_numpy
    
    climber = HillClimber(
        data=data_input,
        objective_func=objective_func,
        max_time=max_time,
        step_size=step_size,
        perturb_fraction=perturb_fraction,
        temperature=temperature,
        cooling_rate=cooling_rate,
        mode=mode,
        target_value=target_value
    )
    
    return climber.climb()
