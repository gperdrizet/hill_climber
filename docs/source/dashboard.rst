Real-time monitoring dashboard
==============================

Hill Climber includes a real-time monitoring dashboard for visualizing optimization progress. Database logging is **enabled by default**, so you can monitor any optimization run without additional configuration.

Features
--------

The dashboard provides:

- **Replica leaderboard**: Top-performing replicas with current objectives, steps, and temperatures
- **Progress statistics**: 
  - Exploration rate (total perturbations per second across all replicas)
  - Progress rate (accepted steps per second)
  - Acceptance rate percentage
- **Interactive time series plots**: Plotly charts for metrics over time with zoom and pan
- **Temperature exchange markers**: Optional visualization of replica exchange events
- **Configurable refresh rate**: Adjust polling frequency (0.5-5 minutes)
- **Plot options**: 
  - Metric selection (Best vs Current history)
  - Normalization toggle
  - Layout control (1 or 2 columns)
  - Downsampling for performance
- **Run information**: Objective function name, dataset size, hyperparameters, and initial temperatures

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

Launching the dashboard
^^^^^^^^^^^^^^^^^^^^^^^

While your optimization is running (or after it completes), launch the dashboard:

.. code-block:: bash

   hill-climber-dashboard

Then navigate to http://localhost:8501 in your browser. The dashboard will automatically discover databases in common locations, or you can select a specific database file using the sidebar controls.

Dashboard configuration
^^^^^^^^^^^^^^^^^^^^^^^

Configure the dashboard using the sidebar:

- **Database selection**: Choose directory and database file from dropdowns
- **Auto-refresh**: Enable/disable automatic updates
- **Refresh interval**: Set polling frequency (0.5-5 minutes)
- **History type**: Select Best (monotonic improvement) or Current (includes exploration)
- **Additional metrics**: Select extra metrics beyond the objective to plot
- **Normalize**: Toggle metric normalization to [0, 1]
- **Exchange markers**: Show vertical lines at temperature exchange events
- **Plot layout**: Choose 1 or 2 column display

Database configuration parameters
----------------------------------

The database logging system uses an efficient collection and write strategy:

- **Worker processes** collect metrics at regular intervals during optimization
- **Main process** performs all database writes after each batch, avoiding lock contention
- No buffering needed - workers return collected metrics to main process

db_enabled : bool, default=True
    Database logging for dashboard monitoring. Enabled by default. Set to ``False`` to disable
    if you don't need real-time monitoring and want to minimize I/O

db_path : str, optional
    Path to SQLite database file. Defaults to ``'../data/hill_climb.db'``

db_step_interval : int, optional
    Collect metrics every Nth step. Uses tiered sampling based on exchange_interval:
    
    - exchange_interval < 10: sample every step (1)
    - exchange_interval 10-99: sample every 10 steps
    - exchange_interval 100-999: sample every 100 steps  
    - exchange_interval >= 1000: sample every 1000 steps
    
    Must be less than or equal to exchange_interval to ensure at least one collection per batch.

checkpoint_interval : int, default=1
    Number of batches between checkpoint saves. Default is 1 (checkpoint every batch).
    Set to higher values (e.g., 10) to reduce checkpoint I/O while database provides
    real-time monitoring

Performance tuning
------------------

Default settings (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       exchange_interval=100,
       # Database logging enabled by default
       # db_step_interval defaults to 100 (one sample per batch)
   )

This provides good balance between resolution and performance:

- Collects 100 steps per replica per batch (100 / 1)
- Main process writes all collected metrics once per batch
- No worker I/O contention

Higher resolution (more database load)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       exchange_interval=1000,
       db_step_interval=100  # Collect every 100th step instead of default 1000
   )

Collects 10 samples per replica per batch instead of 1 (10x higher resolution).

Disabling database logging
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't need real-time monitoring and want to minimize I/O overhead:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective,
       db_enabled=False  # Disable database logging
   )

Database schema
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

Checkpoint independence
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
       db_path='optimization.db'  # Monitor every batch
   )

Complete example
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
   
   # Create optimizer (database logging enabled by default)
   climber = HillClimber(
       data=data,
       objective_func=objective,
       max_time=30,
       n_replicas=4,
       exchange_interval=100,
       db_path='correlation_opt.db',
       checkpoint_file='correlation_opt.pkl',
       checkpoint_interval=5  # Checkpoint every 5 batches
   )
   
   # Run optimization
   best_data = climber.climb()

Then in a separate terminal:

.. code-block:: bash

   hill-climber-dashboard

The dashboard will automatically discover the ``correlation_opt.db`` database file, or you can select it using the sidebar controls.

Troubleshooting
---------------

Database file not found
^^^^^^^^^^^^^^^^^^^^^^^

- Check that the database path in the dashboard matches your configuration
- Verify the optimization has started and completed at least one batch
- Ensure ``db_enabled`` has not been set to ``False``

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
- For I/O-constrained systems, disable database logging with ``db_enabled=False``

Database size estimation
------------------------

With default settings:

- ``exchange_interval=100``
- ``n_replicas=8``
- 20 metrics
- ``db_step_interval=100`` (default: one sample per batch)

Results in:

- 100 steps collected per replica per batch
- 16,000 metric rows per batch
- For 30-minute run (~1000 batches): ~16M rows → 1-2GB database

To reduce size, increase ``db_step_interval``:

.. code-block:: python

   # For exchange_interval >= 1000, only 1 sample per batch by default
   exchange_interval = 5000
   # db_step_interval defaults to 1000 (one sample per 5 batches)

See also
--------

- :doc:`user_guide`: Core optimization concepts
- :doc:`api`: Complete API reference
- :doc:`advanced`: Advanced features and customization
