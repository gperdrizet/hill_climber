User Guide
==========

This guide explains the key concepts and parameters of Hill Climber.

Optimization Modes
------------------

Hill Climber supports two modes:

**Maximize Mode** (``mode='maximize'``)
   Searches for solutions that maximize the objective function value.
   Use this when higher objective values are better.

**Minimize Mode** (``mode='minimize'``)
   Searches for solutions that minimize the objective function value.
   Use this when lower objective values are better.

Objective Functions
-------------------

An objective function takes the data columns as arguments and returns:

1. A dictionary of metrics to track
2. A single objective value to optimize

Hill Climber supports n-dimensional data. Your objective function should accept
as many arguments as you have columns in your input data.

Examples:

.. code-block:: python

   # For 2D data (two columns)
   def objective_2d(x, y):
       # Calculate metrics
       mean_x = np.mean(x)
       mean_y = np.mean(y)
       
       # Calculate objective (e.g., minimize difference)
       objective = -abs(mean_x - mean_y)
       
       # Return metrics and objective
       metrics = {
           'Mean X': mean_x,
           'Mean Y': mean_y,
           'Difference': abs(mean_x - mean_y)
       }
       return metrics, objective

   # For 3D data (three columns)
   def objective_3d(x, y, z):
       correlation_xy = pearsonr(x, y)[0]
       correlation_xz = pearsonr(x, z)[0]
       
       objective = correlation_xy + correlation_xz
       
       metrics = {
           'Corr XY': correlation_xy,
           'Corr XZ': correlation_xz
       }
       return metrics, objective

Hyperparameters
---------------

**step_spread** (default: 1.0)
   Standard deviation of the normal distribution used for perturbations. Controls
   the variability and typical magnitude of changes. Perturbations are sampled
   from a normal distribution with mean 0 and this standard deviation. Larger 
   values create more dramatic perturbations, smaller values make more subtle 
   adjustments.

**perturb_fraction** (default: 0.1)
   Fraction of data points to modify in each iteration (0.0 to 1.0). 
   Higher values create more dramatic changes per step.

**temperature** (default: 1000)
   Initial temperature for simulated annealing. Higher temperatures allow
   more exploration of suboptimal solutions early on.

**cooling_rate** (default: 0.000001)
   Amount subtracted from 1 to get the multiplicative cooling factor. The temperature
   is multiplied by (1 - cooling_rate) each iteration. Smaller values result in slower
   cooling and longer exploration. For example, 0.000001 means temp *= 0.999999 each step.

**max_time** (default: 30)
   Maximum optimization time in minutes.

**initial_noise** (default: 0.0)
   Amount of uniform noise to add when creating replicate starting points.
   Only used in ``climb_parallel()``.

**plot_progress** (default: None)
   Interval in minutes for plotting optimization progress during a run.
   When set, creates scatter plots showing the current best solution at
   regular intervals. For example, ``plot_progress=5`` plots every 5 minutes.
   
   .. note::
      This option only works in single-process mode (``climb()``). It does not
      work with parallel mode (``climb_parallel()``) because results from worker
      processes are not collected until the end of the run.

Boundary Handling
-----------------

Hill Climber uses **reflection** to keep perturbed values within the original
data bounds:

- When a perturbation would push a value beyond the minimum bound, it reflects
  back into the valid range
- Same for maximum bounds
- This prevents artificial accumulation of points at boundaries

Example: If minimum is 5 and a perturbation creates 4.5, it reflects to 5.5.

Replicate Noise
---------------

When running parallel replicates, ``initial_noise`` adds diversity:

.. code-block:: python

   results = climber.climb_parallel(
       replicates=8,
       initial_noise=0.5  # Add ±50% uniform noise to starting data
   )

- Each replicate starts from a slightly different position
- Helps explore different regions of the solution space
- Increases chances of finding global optima
- Generates diverse solutions for comparison

Checkpointing
-------------

For long optimizations, save intermediate progress:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       max_time=60,
       checkpoint_file='optimization.pkl',
       save_interval=300  # Save every 5 minutes
   )
   
   result = climber.climb()

Resume from a checkpoint:

.. code-block:: python

   resumed = HillClimber.resume_from_checkpoint(
       checkpoint_file='optimization.pkl',
       objective_func=my_objective,
       new_max_time=30  # Continue for 30 more minutes
   )
   
   result = resumed.climb()

Results Structure
-----------------

**Single climb** returns a tuple of:

.. code-block:: python

   best_data, steps_df = climber.climb()

Where:

- ``best_data``: Optimized data (DataFrame or numpy array, same format as input)
- ``steps_df``: DataFrame tracking optimization progress with columns:
  
  - ``Step``: Step number when improvement was accepted
  - ``Objective value``: Objective value at that step
  - ``Best_data``: Snapshot of best data at that step
  - Additional metric columns (defined by your objective function)

**Parallel climbs** returns a dictionary:

.. code-block:: python

   results = climber.climb_parallel(replicates=4)
   
   # Structure:
   {
       'input_data': <original DataFrame>,
       'results': [
           (initial_data_1, best_data_1, steps_df_1),
           (initial_data_2, best_data_2, steps_df_2),
           ...
       ]
   }

Where:

- ``input_data``: Original data before any noise was added
- ``initial_data``: Data for this replicate after noise addition
- ``best_data``: Final optimized data for this replicate
- ``steps_df``: Optimization history for this replicate

Internal Architecture
---------------------

Hill Climber uses a unified ``OptimizerState`` dataclass to manage all optimization
progress internally. This provides:

- **Clean separation**: Hyperparameters stay in ``HillClimber``, runtime state in ``OptimizerState``
- **Easy checkpointing**: State can be serialized/deserialized as a unit
- **Better organization**: All tracking data (current/best solutions, metrics, history, timing) in one place
- **Type safety**: Dataclass provides clear typing for all state attributes

You don't need to interact with ``OptimizerState`` directly - it's used internally
by the ``HillClimber`` class. However, if you're extending or debugging the code,
you can access it via ``climber.state``.
