User guide
==========

This guide explains the key concepts and parameters of Hill Climber.

Data format and terminology
----------------------------

Hill Climber works with tabular data:

**Data Shape**
   Input data has shape ``(N, M)`` where:
   
   - ``N`` = number of samples (rows/data points)
   - ``M`` = number of features (columns)

**Accepted formats**
   - NumPy arrays: ``np.ndarray`` with shape ``(N, M)``
   - Pandas DataFrames: ``pd.DataFrame`` with M columns

**Objective function signature**
   Your objective function receives M separate 1D arrays (one per column):
   
   - For M=2: ``objective_func(x, y)``
   - For M=3: ``objective_func(x, y, z)``
   - For M=4: ``objective_func(w, x, y, z)``

.. note::
   The term "2D data" in this documentation refers to data with 2 features (M=2),
   not the numpy array dimensionality. All input data are 2D numpy arrays with
   shape ``(N, M)``.

Optimization modes
------------------

Hill Climber supports three modes:

**Maximize mode** (``mode='maximize'``)
   Searches for solutions that maximize the objective function value.
   Use this when higher objective values are better.

**Minimize mode** (``mode='minimize'``)
   Searches for solutions that minimize the objective function value.
   Use this when lower objective values are better.

**Target mode** (``mode='target'``)
   Searches for solutions that approach a specific target value.
   Requires setting ``target_value`` parameter. The objective function
   should return the distance from the target (minimized internally).

Objective functions
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

      # Calculate metrics
      correlation_xy = pearsonr(x, y)[0]
      correlation_xz = pearsonr(x, z)[0]
       
      # Calculate objective (e.g., maximize sum of correlations)
      objective = correlation_xy + correlation_xz
      
      # Return metrics and objective
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

**T_min** (default: 0.0001)
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

**exchange_interval** (default: 100)
   Number of optimization steps between replica exchange attempts. Smaller values
   attempt exchanges more frequently but increase overhead.

**exchange_strategy** (default: 'even_odd')
   Strategy for selecting replica pairs for exchange:
   
   - 'even_odd': Alternates between even and odd neighboring pairs
   - 'random': Random pair selection
   - 'all_neighbors': All neighboring pairs

**initial_step_spread** (default: 0.25)
   Initial perturbation spread as a fraction of each feature's data range (0.25 = 25% of range).
   Controls the magnitude of changes relative to your data scale. The actual perturbation
   standard deviation is calculated per-feature as ``initial_step_spread * feature_range``,
   where each feature uses its own range for more appropriate perturbations across different
   scales. Larger values create more dramatic perturbations, smaller values make more subtle
   adjustments.

**final_step_spread** (default: None)
   Final perturbation spread as a fraction of each feature's data range. If specified,
   step spread decreases from initial_step_spread to final_step_spread over
   the course of max_time, enabling time-based cooling for more refined optimization
   near the end of the run. Leave as None to maintain constant step spread throughout.

**step_spread_scheme** (default: 'linear')
   Controls how step spread decreases over time when ``final_step_spread`` is specified:
   
   - 'linear': Steady decrease from initial to final
   - 'geometric': Exponential decay, spending more time at smaller step sizes
   - 'zeno': Halve at t=0.5, then t=0.75, t=0.875, etc. (Zeno's paradox schedule)

**perturb_fraction** (default: 0.001)
   Fraction of data points to modify in each iteration (0.0 to 1.0). 
   Higher values create more dramatic changes per step.

**cooling_rate** (default: 1e-10)
   Amount subtracted from 1 to get the multiplicative cooling factor. The temperature
   is multiplied by ``(1 - cooling_rate)`` each iteration. Smaller values result in slower
   cooling and longer exploration. For example, ``1e-10`` means ``temp *= 0.9999999999`` each step.

**max_time** (default: 10)
   Maximum optimization time in minutes.

**checkpoint_file** (default: None)
   Path to save checkpoints. If specified, the optimizer saves its state after each
   batch, allowing resumption if interrupted.

**checkpoint_interval** (default: 1)
   Number of batches between checkpoint saves. Default is 1 (save every batch).
   Set higher to reduce I/O overhead.

**db_enabled** (default: True)
   Enable database logging for real-time dashboard monitoring. Requires installing
   with dashboard extras.

**db_path** (default: '../data/hill_climb.db')
   Path to SQLite database file for dashboard data.

**db_step_interval** (default: tiered based on exchange_interval)
   Snapshot interval for database logging. Every Nth step, records the current state:
   
   - Current perturbation being evaluated
   - Current accepted state (if changed)
   - Current best solution (if changed)
   
   Uses tiered sampling:
   
   - exchange_interval < 10: snapshot every step (db_step_interval = 1)
   - exchange_interval 10-99: snapshot every 10 steps (db_step_interval = 10)
   - exchange_interval 100-999: snapshot every 100 steps (db_step_interval = 100)
   - exchange_interval >= 1000: snapshot every 1000 steps (db_step_interval = 1000)
   
   This creates a regularly-sampled view of optimization progress while keeping
   database size manageable.

**verbose** (default: False)
   Print progress messages during optimization.

**n_workers** (default: n_replicas)
   Number of worker processes for parallel execution. Defaults to number of replicas.

Checkpoints
-----------

If you specify a ``checkpoint_file`` path, the optimizer saves its state periodically,
allowing you to resume from the most recent state if interrupted or continue optimization
after the run ends.

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
   best_data = resumed.climb()

.. note::
   It is also possible to resume a run while it is still in memory by simply calling
   ``climb()`` again on the existing ``HillClimber`` instance.

**Batch size**
   The batch size is determined by ``exchange_interval`` (default: 100 steps).
   After each batch, the optimizer:
   
   - Attempts replica exchanges
   - Saves a checkpoint (if ``checkpoint_file`` is specified and checkpoint_interval condition is met)
   - Updates the progress dashboard database (if ``db_enabled`` is True)

**Checkpoint frequency**
   The actual frequency of checkpoints is controlled by ``checkpoint_interval``. By default,
   a checkpoint is saved after every batch (i.e., every ``exchange_interval`` steps). You can
   save checkpoints less frequently by setting ``checkpoint_interval`` to a higher value to reduce I/O.

Boundary handling
-----------------

Hill climber uses **reflection** to keep perturbed values within the original
data bounds:

- When a perturbation would push a value beyond the minimum bound, it reflects
  back into the valid range
- Same for maximum bounds
- This prevents artificial accumulation of points at boundaries

Example: If minimum is 5 and a perturbation creates 4.5, it reflects to 5.5.

Replica exchange (parallel tempering)
-------------------------------------

Hill climber uses replica exchange to improve global optimization. Multiple
replicas run simultaneously at different temperatures:

**How it works**

1. Each replica has its own temperature from a ladder (e.g., 1000, 2154, 4641, 10000)
2. All replicas perform optimization steps independently
3. Periodically, replicas attempt to exchange configurations
4. Exchanges use Metropolis criterion: better solutions move to cooler temperatures
5. The coldest replica typically finds the best solution

**Temperature ladder**

.. code-block:: python

   from hill_climber import TemperatureLadder
   
   # Geometric spacing (default, recommended)
   ladder = TemperatureLadder.geometric(n_replicas=4, T_min=0.0001, T_max=0.01)
   print(ladder.temperatures)  # [0.0001, 0.000215, 0.000464, 0.01]
   
   # Linear spacing
   ladder = TemperatureLadder.linear(n_replicas=4, T_min=0.0001, T_max=0.01)
   print(ladder.temperatures)  # [0.0001, 0.0034, 0.0067, 0.01]

**Benefits**

- Better global optimization compared to single-temperature annealing
- Hotter replicas explore broadly, cooler replicas exploit locally
- Exchanges allow good solutions to refine at low temperatures
- More robust than independent parallel runs

Results structure
-----------------

The ``climb()`` method returns the best data found:

.. code-block:: python

   best_data = climber.climb()

Where:

- ``best_data``: Optimized data (DataFrame or numpy array, same format as input) from the best-performing replica

After optimization, you can access replica state:

.. code-block:: python

   # Get best replica
   best_replica = max(climber.replicas, key=lambda r: r['best_objective'])
   
   # Access state
   print(f"Perturbation number: {best_replica['perturbation_num']}")
   print(f"Accepted steps: {best_replica['num_accepted']}")
   print(f"Improvements found: {best_replica['num_improvements']}")
   print(f"Best objective: {best_replica['best_objective']}")
   print(f"Best metrics: {best_replica['best_metrics']}")

State tracking
--------------

Hill Climber uses a snapshot-based approach to track optimization state:

**Snapshot recording** (every db_step_interval steps)
   At regular intervals, the optimizer records three snapshots:
   
   1. **Current perturbation**: The perturbation being evaluated at this step
   2. **Current accepted state**: The currently accepted solution (if changed since last snapshot)
   3. **Current best state**: The current best solution found (if changed since last snapshot)

**Three views of optimization**
   - **Perturbations**: Shows what's being explored (all attempts)
   - **Accepted steps**: Shows the simulated annealing trajectory (SA path)
   - **Improvements**: Shows monotonic progress toward optimal solution

All events are indexed by ``perturbation_num``, a monotonically increasing counter
that never resets. This provides:

- **Compact storage**: Only snapshots at regular intervals, not every event
- **Three perspectives**: Exploration, acceptance, and improvement
- **Clear semantics**: Single time axis across all views
- **Efficient queries**: Regular sampling enables fast dashboard updates

Database schema
---------------

When ``db_enabled=True``, Hill Climber creates:

**perturbations** table
   Snapshots of current perturbation at each db_step_interval
   
   - replica_id, perturbation_num, objective, is_accepted, is_improvement, temperature

**accepted_steps** table
   Snapshots of current accepted state (when changed since last snapshot)
   
   - replica_id, perturbation_num, objective, temperature

**step_metrics** table
   User-defined metrics for accepted state snapshots
   
   - replica_id, perturbation_num, metric_name, value

**improvements** table
   Snapshots of current best solution (when changed since last snapshot)
   
   - replica_id, perturbation_num, best_objective, temperature

**improvement_metrics** table
   User-defined metrics for best solution snapshots
   
   - replica_id, perturbation_num, metric_name, value

**replica_status** table
   Current snapshot of each replica (updated after each batch)
   
   - replica_id, current_perturbation_num, num_accepted, num_improvements, best_objective, temperature

Internal Architecture
---------------------

Hill Climber uses a ``ReplicaState`` dataclass to manage the state of each replica
during optimization. Key state attributes:

- **perturbation_num**: Global counter (increments with every evaluation)
- **num_accepted**: Count of SA-accepted moves  
- **num_improvements**: Count of improvements found
- **current_data**: Current solution being explored
- **best_data**: Best solution found so far
- **best_objective**: Best objective value found
- **best_metrics**: User-defined metrics at best solution
- **temperature**: Current temperature

This provides:

- **Clean separation**: Hyperparameters in ``HillClimber``, runtime state in ``ReplicaState``
- **Easy checkpointing**: State serializes as a unit
- **Single counter**: No confusion between different metrics
- **Type safety**: Dataclass provides clear typing

You don't need to interact with ``ReplicaState`` directly - it's used internally
by the ``HillClimber`` class to manage each replica's optimization state.
