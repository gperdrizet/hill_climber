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

Hill Climber supports three modes:

**Maximize Mode** (``mode='maximize'``)
   Searches for solutions that maximize the objective function value.
   Use this when higher objective values are better.

**Minimize Mode** (``mode='minimize'``)
   Searches for solutions that minimize the objective function value.
   Use this when lower objective values are better.

**Target Mode** (``mode='target'``)
   Searches for solutions that approach a specific target value.
   Requires setting ``target_value`` parameter. The objective function
   should return the distance from the target (minimized internally).

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
   temperature from the temperature ladder. Set to 1 for simulated annealing without
   replica exchange.

**T_min** (default: 0.1)
   Minimum temperature for the coldest replica in the temperature ladder.
   Also used as the base temperature for simulated annealing. Higher temperatures
   allow more exploration of suboptimal solutions.

**T_max** (default: 100 * T_min)
   Maximum temperature for the hottest replica in the temperature ladder. Should
   be significantly higher than T_min for effective replica exchange. Defaults to
   100 times T_min if not specified.

**temperature_scheme** (default: 'geometric')
   How to space temperatures in the ladder: 'geometric' or 'linear'. Geometric
   spacing typically provides better exchange acceptance rates.

**exchange_interval** (default: 10000)
   Number of optimization steps between replica exchange attempts. Smaller values
   attempt exchanges more frequently but increase overhead.

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

**perturb_fraction** (default: 0.001)
   Fraction of data points to modify in each iteration (0.0 to 1.0). 
   Higher values create more dramatic changes per step.

**cooling_rate** (default: 1e-8)
   Amount subtracted from 1 to get the multiplicative cooling factor. The temperature
   is multiplied by ``(1 - cooling_rate)`` each iteration. Smaller values result in slower
   cooling and longer exploration. For example, ``1e-8`` means ``temp *= 0.99999999`` each step.

**max_time** (default: 30)
   Maximum optimization time in minutes.

**checkpoint_file** (default: None)
   Path to save checkpoints. If specified, the optimizer saves its state after each
   batch, allowing resumption if interrupted.

**checkpoint_interval** (default: 1)
   Number of batches between checkpoint saves. Default is 1 (save every batch).
   Set higher to reduce I/O overhead.

**db_enabled** (default: False)
   Enable database logging for real-time dashboard monitoring. Requires installing
   with dashboard extras.

**db_path** (default: 'data/hill_climber_progress.db')
   Path to SQLite database file for dashboard data.

**db_step_interval** (default: exchange_interval // 10)
   Collect metrics every Nth step for database logging. Defaults to 10% sampling
   (every 10th step). For small exchange intervals (≤10), defaults to 1 (every step).
   Must be less than exchange_interval.

**verbose** (default: False)
   Print progress messages during optimization.

**n_workers** (default: n_replicas)
   Number of worker processes for parallel execution. Defaults to number of replicas.

Checkpoints
-----------

If you specify a ``checkpoint_file`` path, the optimizer saves its state periodically,
allowing you to resume from the most recent state if interrupted or continue optimization
after the run ends.

.. note::
   Checkpoints store the entire optimizer state, including current solutions,
   best solutions, temperatures, and history. This allows seamless resumption.

**Batch Size**
   The batch size is determined by ``exchange_interval`` (default: 10000 steps).
   After each batch, the optimizer:
   
   - Attempts replica exchanges
   - Saves a checkpoint (if ``checkpoint_file`` is specified and checkpoint_interval condition is met)
   - Updates the progress dashboard database (if ``db_enabled`` is True)

**Checkpoint Frequency**
   The actual frequency of checkpoints is controlled by ``checkpoint_interval``. By default,
   a checkpoint is saved after every batch (i.e., every ``exchange_interval`` steps). You can
   save checkpoints less frequently by setting ``checkpoint_interval`` to a higher value to reduce I/O.

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
-------------------------------------

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

.. note::
   It is also possible to resume a run while it is still in memory by simply calling
   ``climb()`` again on the existing ``HillClimber`` instance.

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

Hill Climber uses a ``ReplicaState`` dataclass to manage the state of each replica
during optimization. This provides:

- **Clean separation**: Hyperparameters stay in ``HillClimber``, runtime state in ``ReplicaState``
- **Easy checkpointing**: State can be serialized/deserialized as a unit
- **Better organization**: All tracking data (current/best solutions, metrics, history, timing) in one place
- **Type safety**: Dataclass provides clear typing for all state attributes

You don't need to interact with ``ReplicaState`` directly - it's used internally
by the ``HillClimber`` class to manage each replica's optimization state.
