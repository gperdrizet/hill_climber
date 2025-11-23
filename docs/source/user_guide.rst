User Guide
==========

This guide explains the key concepts and parameters of Hill Climber.

Data Format and Terminology
----------------------------

Hill Climber works with tabular data:

**Data Shape**
   Input data has shape ``(N, M)`` where:
   
   - ``N`` = number of samples (rows/data points)
   - ``M`` = number of features (columns)

**Accepted Formats**
   - NumPy arrays: ``np.ndarray`` with shape ``(N, M)``
   - Pandas DataFrames: ``pd.DataFrame`` with M columns

**Objective Function Signature**
   Your objective function receives M separate 1D arrays (one per column):
   
   - For M=2: ``objective_func(x, y)``
   - For M=3: ``objective_func(x, y, z)``
   - For M=4: ``objective_func(w, x, y, z)``

.. note::
   The term "2D data" in this documentation refers to data with 2 features (M=2),
   not the numpy array dimensionality. All input data are 2D numpy arrays with
   shape ``(N, M)``.

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

Your objective function should accept as many arguments as you have columns
in your input data.

Examples:

.. code-block:: python

   # For 2-column data (M=2)
   def objective_2col(x, y):
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

   # For 3-column data (M=3)
   def objective_3col(x, y, z):
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

**n_replicas** (default: 4)
   Number of replicas for parallel tempering (replica exchange). More replicas
   provide better exploration but use more memory. Each replica runs at a different
   temperature from the temperature ladder.

**temperature** (default: 1000)
   Minimum temperature (T_min) for the coldest replica in the temperature ladder.
   Also used as the base temperature for simulated annealing. Higher temperatures
   allow more exploration of suboptimal solutions.

**T_max** (default: 10000)
   Maximum temperature for the hottest replica in the temperature ladder. Should
   be significantly higher than T_min for effective replica exchange.

**temperature_scheme** (default: 'geometric')
   How to space temperatures in the ladder: 'geometric' or 'linear'. Geometric
   spacing typically provides better exchange acceptance rates.

**exchange_interval** (default: 100)
   Number of optimization steps between replica exchange attempts. Smaller values
   attempt exchanges more frequently.

**exchange_strategy** (default: 'even_odd')
   Strategy for selecting replica pairs for exchange:
   
   - 'even_odd': Alternates between even and odd neighboring pairs
   - 'random': Random pair selection
   - 'all_neighbors': All neighboring pairs

**step_spread** (default: 0.01)
   Perturbation spread as a fraction of the input data range (0.01 = 1% of range).
   Controls the magnitude of changes relative to your data scale. The actual perturbation
   standard deviation is calculated as ``step_spread * mean(data_range)``, making it
   automatically scale-appropriate for your data. Larger values create more dramatic
   perturbations, smaller values make more subtle adjustments.

**perturb_fraction** (default: 0.1)
   Fraction of data points to modify in each iteration (0.0 to 1.0). 
   Higher values create more dramatic changes per step.

**cooling_rate** (default: 0.000001)
   Amount subtracted from 1 to get the multiplicative cooling factor. The temperature
   is multiplied by ``(1 - cooling_rate)`` each iteration. Smaller values result in slower
   cooling and longer exploration. For example, ``0.000001`` means ``temp *= 0.999999`` each step.

**max_time** (default: 30)
   Maximum optimization time in minutes.

**show_progress** (default: True)
   Whether to display progress plots during optimization. Set to False to disable
   plots (useful for automated scripts or testing).

Checkpoints and Progress Plotting
----------------------------------

Hill Climber automatically saves checkpoints and updates progress plots after each
batch of optimization steps (controlled by ``exchange_interval``). If you specify a
``checkpoint_file`` path, the optimizer saves its state after every batch, allowing
you to resume from the most recent state if interrupted. Progress plots are shown
after each batch when ``show_progress=True`` (default).

**Batch Size**
   The batch size is determined by ``exchange_interval`` (default: 100 steps).
   After each batch, the optimizer:
   
   - Attempts replica exchanges
   - Saves a checkpoint (if ``checkpoint_file`` is specified)
   - Updates progress plots (if ``show_progress=True``)

Boundary Handling
-----------------

Hill Climber uses **reflection** to keep perturbed values within the original
data bounds:

- When a perturbation would push a value beyond the minimum bound, it reflects
  back into the valid range
- Same for maximum bounds
- This prevents artificial accumulation of points at boundaries

Example: If minimum is 5 and a perturbation creates 4.5, it reflects to 5.5.

Replica Exchange (Parallel Tempering)
--------------------------------------

Hill Climber 2.0 uses replica exchange to improve global optimization. Multiple
replicas run simultaneously at different temperatures:

**How it works:**

1. Each replica has its own temperature from a ladder (e.g., 1000, 2154, 4641, 10000)
2. All replicas perform optimization steps independently
3. Periodically, replicas attempt to exchange configurations
4. Exchanges use Metropolis criterion: better solutions move to cooler temperatures
5. The coldest replica typically finds the best solution

**Temperature Ladder:**

.. code-block:: python

   from hill_climber import TemperatureLadder
   
   # Geometric spacing (default, recommended)
   ladder = TemperatureLadder.geometric(n_replicas=4, T_min=1000, T_max=10000)
   print(ladder.temperatures)  # [1000, 2154, 4641, 10000]
   
   # Linear spacing
   ladder = TemperatureLadder.linear(n_replicas=4, T_min=1000, T_max=10000)
   print(ladder.temperatures)  # [1000, 4000, 7000, 10000]

**Exchange Statistics:**

After optimization, check exchange acceptance rates:

.. code-block:: python

   best_data, history_df = climber.climb()
   
   # Exchange statistics are printed during optimization
   # Exchange acceptance rate: 15.4%
   
   # For detailed analysis, access the climber's exchange statistics:
   # climber.exchange_stats.get_acceptance_matrix()

**Benefits:**

- Better global optimization compared to single-temperature annealing
- Hotter replicas explore broadly, cooler replicas exploit locally
- Exchanges allow good solutions to refine at low temperatures
- More robust than independent parallel runs

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

   # Continue with saved temperatures (default)
   resumed = HillClimber.load_checkpoint(
       filepath='optimization.pkl',
       objective_func=my_objective
   )
   
   # Or reset temperatures to original ladder values
   resumed = HillClimber.load_checkpoint(
       filepath='optimization.pkl',
       objective_func=my_objective,
       reset_temperatures=True
   )
   
   # Continue optimizing
   best_data, history_df = resumed.climb()

Results Structure
-----------------

The ``climb()`` method returns a tuple:

.. code-block:: python

   best_data, steps_df = climber.climb()

Where:

- ``best_data``: Optimized data (DataFrame or numpy array, same format as input)
- ``steps_df``: DataFrame tracking optimization progress with columns:
  
  - ``Step``: Step number when improvement was accepted
  - ``Objective value``: Objective value at that step
  - ``Best_data``: Snapshot of best data at that step
  - Additional metric columns (defined by your objective function)

The best result is automatically selected from the replica with the best objective value.

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
