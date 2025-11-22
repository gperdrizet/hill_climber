Quick Start
===========

This guide will get you started with Hill Climber in just a few minutes.

Hill Climber works with multi-column datasets. Your objective function receives
one argument for each column/feature in your data.

Basic Example
-------------

Here's a simple example that optimizes a 2-column dataset for high Pearson correlation:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from hill_climber import HillClimber
   from scipy.stats import pearsonr

   # Create initial random data
   data = pd.DataFrame({
       'x': np.random.rand(1000),
       'y': np.random.rand(1000)
   })

   # Define objective function
   def objective_high_correlation(x, y):
       """Maximize Pearson correlation."""
       corr = pearsonr(x, y)[0]
       metrics = {'Pearson correlation': corr}
       return metrics, abs(corr)

   # Create optimizer with replica exchange
   climber = HillClimber(
       data=data,
       objective_func=objective_high_correlation,
       max_time=5,  # 5 minutes
       mode='maximize',
       n_replicas=4  # Use 4 replicas for parallel tempering
   )

   # Run optimization
   best_data, steps_df = climber.climb()

   # View results
   print(f"Final correlation: {steps_df['Pearson correlation'].iloc[-1]:.3f}")

Monitoring Progress
-------------------

For longer runs, monitor progress with live plots:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective_high_correlation,
       max_time=30,
       mode='maximize',
       plot_progress=5  # Plot every 5 minutes
   )

   best_data, steps_df = climber.climb()

Understanding the Results
--------------------------

The ``climb()`` method returns a tuple of ``(best_data, steps_df)``:

- ``best_data``: The optimized data (same format as input - DataFrame or numpy array)
- ``steps_df``: A DataFrame tracking the optimization history at each accepted step,
  including the objective value and all metrics you defined

Replica Exchange (Parallel Tempering)
--------------------------------------

Hill Climber 2.0 uses replica exchange (parallel tempering) by default. Multiple
replicas run at different temperatures and exchange configurations to improve
global optimization:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective_high_correlation,
       max_time=10,
       mode='maximize',
       n_replicas=8,  # Number of replicas (default: 4)
       temperature=1000,  # Minimum temperature (T_min)
       T_max=10000,  # Maximum temperature
       exchange_interval=100,  # Steps between exchange attempts
       temperature_scheme='geometric'  # or 'linear'
   )

   best_data, steps_df = climber.climb()

The ``climb()`` method automatically runs all replicas and returns the best result.

Visualization
-------------

Visualize the optimization progress:

.. code-block:: python

   # Visualize single result
   climber.plot_results(
       (best_data, steps_df),
       metrics=['Pearson correlation'],
       plot_type='histogram'
   )

Next Steps
----------

- See :doc:`user_guide` for detailed explanations of hyperparameters
- Check :doc:`notebooks` for complete examples
- Explore :doc:`api` for full API reference
