Quick Start
===========

This guide will get you started with Hill Climber in just a few minutes.

.. note::
   Hill Climber currently supports a maximum of 2D input data (two columns: x and y).

Basic Example
-------------

Here's a simple example that creates a dataset with high Pearson correlation:

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
   result = climber.climb()

   # View results
   print(f"Final correlation: {result['Pearson correlation']:.3f}")

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

The ``climb()`` method returns the optimized data and a history of metrics at each step.

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
