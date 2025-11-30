Real-Time Monitoring Dashboard
==============================

The Hill Climber package includes a real-time monitoring dashboard built with Streamlit and SQLite for visualizing optimization progress as it runs. The dashboard uses a modular architecture separating data loading, UI components, and plot generation.

Features
--------

The dashboard provides:

- **Replica Leaderboard**: Top-performing replicas with current objectives, steps, and temperatures
- **Progress Statistics**: 
  - Exploration rate (total perturbations per second across all replicas)
  - Progress rate (accepted steps per second)
  - Acceptance rate percentage
- **Interactive Time Series Plots**: Plotly charts for metrics over time with zoom and pan
- **Temperature Exchange Markers**: Optional visualization of replica exchange events
- **Configurable Refresh Rate**: Adjust polling frequency (0.5-5 minutes)
- **Plot Options**: 
  - Metric selection (Best vs Current history)
  - Normalization toggle
  - Layout control (1 or 2 columns)
  - Downsampling for performance
- **Run Information**: Objective function name, dataset size, hyperparameters, and initial temperatures

Installation
------------

The dashboard requires additional dependencies. Install with dashboard extras:

.. code-block:: bash

   pip install parallel-hill-climber[dashboard]

This will install:

- ``streamlit``: Web dashboard framework
- ``plotly``: Interactive plotting library

Usage
-----

Enabling Database Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the dashboard, enable database logging in your HillClimber instance:

.. code-block:: python

   from hill_climber import HillClimber
   
   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       db_enabled=True,  # Enable database logging
       db_path='my_optimization.db',  # Optional: custom path
       db_step_interval=100,  # Optional: collect every 100th step
       checkpoint_interval=10,  # Optional: checkpoint every 10 batches
       # ... other parameters
   )
   
   best_data, history = climber.climb()

Launching the Dashboard
^^^^^^^^^^^^^^^^^^^^^^^

While your optimization is running (or after it completes), launch the dashboard:

.. code-block:: bash

   hill-climber-dashboard

Then navigate to http://localhost:8501 in your browser. The dashboard will automatically discover databases in common locations, or you can select a specific database file using the sidebar controls.

Dashboard Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Configure the dashboard using the sidebar:

- **Database Selection**: Choose directory and database file from dropdowns
- **Auto-refresh**: Enable/disable automatic updates
- **Refresh Interval**: Set polling frequency (0.5-5 minutes)
- **History Type**: Select Best (monotonic improvement) or Current (includes exploration)
- **Additional Metrics**: Select extra metrics beyond the objective to plot
- **Normalize**: Toggle metric normalization to [0, 1]
- **Exchange Markers**: Show vertical lines at temperature exchange events
- **Plot Layout**: Choose 1 or 2 column display

Database Configuration Parameters
----------------------------------

The database logging system uses an efficient collection and write strategy:

- **Worker processes** collect metrics at regular intervals during optimization
- **Main process** performs all database writes after each batch, avoiding lock contention
- No buffering needed - workers return collected metrics to main process

db_enabled : bool, default=False
    Enable database logging for dashboard monitoring

db_path : str, optional
    Path to SQLite database file. Defaults to ``'data/hill_climber_progress.db'``

db_step_interval : int, optional
    Collect metrics every Nth step. Default: ``exchange_interval // 10`` (10% sampling).
    For small exchange intervals (≤10), defaults to 1 (every step). Must be less than
    exchange_interval to ensure at least one collection per batch.

checkpoint_interval : int, default=1
    Number of batches between checkpoint saves. Default is 1 (checkpoint every batch).
    Set to higher values (e.g., 10) to reduce checkpoint I/O while database provides
    real-time monitoring

Performance Tuning
------------------

Default Settings (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       exchange_interval=10000,
       db_enabled=True,
       # db_step_interval defaults to 10000 // 10 = 1000 (10% sampling)
   )

This provides good balance between resolution and performance:

- Collects 1000 steps per replica per batch (10,000 / 10)
- Main process writes all collected metrics once per batch
- No worker I/O contention

Higher Resolution (More Database Load)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       exchange_interval=10000,
       db_enabled=True,
       db_step_interval=500  # Collect every 500th step (5% sampling)
   )

Collects 2000 steps per replica per batch (twice the default resolution).

Lower Resolution (Faster, Smaller Database)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       exchange_interval=10000,
       db_enabled=True,
       db_step_interval=2000  # Collect every 2000th step (20% sampling)
   )

Collects 500 steps per replica per batch (half the default, smaller database).

Database Schema
---------------

The database contains four tables:

run_metadata
^^^^^^^^^^^^

Stores run configuration and hyperparameters:

- ``run_id``: Always 1 (single run per database)
- ``start_time``: Unix timestamp when optimization started
- ``n_replicas``: Number of replicas
- ``exchange_interval``: Steps between exchange attempts
- ``db_step_interval``: Step collection frequency
- ``hyperparameters``: JSON-encoded hyperparameters

replica_status
^^^^^^^^^^^^^^

Current state of each replica (updated after each batch):

- ``replica_id``: Replica identifier (0 to n_replicas-1)
- ``step``: Current step number
- ``temperature``: Current temperature
- ``best_objective``: Best objective value found
- ``current_objective``: Current objective value
- ``timestamp``: Unix timestamp of last update

metrics_history
^^^^^^^^^^^^^^^

Time series of metrics (sampled according to ``db_step_interval``):

- ``replica_id``: Replica identifier
- ``step``: Step number when metric was recorded
- ``metric_name``: Name of the metric
- ``value``: Metric value

Indexed on ``(replica_id, step)`` for fast queries.

temperature_exchanges
^^^^^^^^^^^^^^^^^^^^^

Record of temperature swaps between replicas:

- ``step``: Step number when exchange occurred
- ``replica_id``: Replica that received new temperature
- ``new_temperature``: New temperature after exchange
- ``timestamp``: Unix timestamp

Checkpoint Independence
-----------------------

Database logging and checkpointing are decoupled for flexibility:

- **Database**: Provides real-time progress monitoring with configurable granularity
- **Checkpoints**: Provide full state recovery with configurable frequency

This allows you to:

- Monitor progress every batch while checkpointing every 10 batches
- Reduce checkpoint file I/O overhead
- Accept partial batch loss on crashes (database provides progress visibility, checkpoints provide recovery)

Example:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       checkpoint_file='optimization.pkl',
       checkpoint_interval=10,  # Checkpoint every 10 batches
       db_enabled=True,
       db_path='optimization.db'  # Monitor every batch
   )

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from hill_climber import HillClimber
   
   # Generate data
   np.random.seed(42)
   data = pd.DataFrame({
       'x': np.random.randn(1000),
       'y': np.random.randn(1000)
   })
   
   # Define objective
   def objective(x, y):
       corr = np.corrcoef(x, y)[0, 1]
       return {'Correlation': corr}, corr
   
   # Create optimizer with database enabled
   climber = HillClimber(
       data=data,
       objective_func=objective,
       max_time=30,
       n_replicas=4,
       exchange_interval=10000,
       db_enabled=True,
       db_path='correlation_opt.db',
       checkpoint_file='correlation_opt.pkl',
       checkpoint_interval=5  # Checkpoint every 5 batches
   )
   
   # Run optimization
   best_data, history = climber.climb()

Then in a separate terminal:

.. code-block:: bash

   streamlit run progress_dashboard.py

Set the database path to ``correlation_opt.db`` in the dashboard sidebar.

Troubleshooting
---------------

Database file not found
^^^^^^^^^^^^^^^^^^^^^^^

- Ensure your HillClimber instance has ``db_enabled=True``
- Check that the database path in the dashboard matches your configuration
- Verify the optimization has started and completed at least one batch

No data appearing
^^^^^^^^^^^^^^^^^

- Wait for the first batch to complete (``exchange_interval`` steps)
- Check the "Run Information" in the sidebar to verify database configuration
- Ensure auto-refresh is enabled or click "Refresh Now"

Slow dashboard updates
^^^^^^^^^^^^^^^^^^^^^^

- Reduce the number of metrics displayed
- Increase the refresh interval
- Increase ``db_step_interval`` to reduce database size

Slow optimization performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Increase ``db_step_interval`` to reduce collection overhead
- Consider disabling database logging (``db_enabled=False``) for production runs
- Use checkpoints for state recovery instead of database monitoring

Database Size Estimation
------------------------

With default settings:

- ``exchange_interval=10000``
- ``n_replicas=8``
- 20 metrics
- ``db_step_interval=1000`` (default: exchange_interval // 10)

Results in:

- 1000 steps collected per replica per batch
- 160,000 metric rows per batch
- For 30-minute run (~100 batches): ~16M rows → 1-2GB database

To reduce size, increase ``db_step_interval``:

.. code-block:: python

   # Half the database size
   db_step_interval = 2000  # 20% sampling instead of 10%

See Also
--------

- :doc:`user_guide`: Core optimization concepts
- :doc:`api`: Complete API reference
- :doc:`advanced`: Advanced features and customization
