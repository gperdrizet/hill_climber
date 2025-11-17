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

   # Create optimizer
   climber = HillClimber(
       data=data,
       objective_func=objective_high_correlation,
       max_time=5,  # 5 minutes
       mode='maximize'
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

   result = climber.climb()

.. note::
   Progress plotting only works with ``climb()`` (not ``climb_parallel()``).

Understanding the Results
--------------------------

The ``climb()`` method returns a tuple of ``(best_data, steps_df)``:

- ``best_data``: The optimized data (same format as input - DataFrame or numpy array)
- ``steps_df``: A DataFrame tracking the optimization history at each accepted step,
  including the objective value and all metrics you defined

Parallel Replicates
-------------------

Run multiple independent optimizations to explore different solutions:

.. code-block:: python

   results = climber.climb_parallel(
       replicates=4,
       initial_noise=0.5
   )

   # Results is a dictionary with:
   # - 'input_data': original data
   # - 'results': list of (initial_data, best_data, steps_df) tuples

Visualization
-------------

Visualize the optimization progress:

.. code-block:: python

   climber.plot_results(
       results,
       metrics=['Pearson correlation'],
       plot_type='histogram'
   )

Next Steps
----------

- See :doc:`user_guide` for detailed explanations of hyperparameters
- Check :doc:`notebooks` for complete examples
- Explore :doc:`api` for full API reference
