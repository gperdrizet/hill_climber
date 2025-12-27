Quick start
===========

This guide will get you started with Hill Climber in just a few minutes.

Hill Climber works with multi-column datasets. Your objective function receives
one argument for each column/feature in your data. The objective function must
return a tuple of (metrics_dict, objective_value), where metrics_dict is a
dictionary of metrics for tracking and objective_value is the scalar to optimize.

Basic example
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
   best_data = climber.climb()

   # View results
   print(f'Best replica: {climber.replicas[0]["replica_id"]}')
   print(f'Best objective: {climber.replicas[0]["best_objective"]:.3f}')
   print(f'Total perturbations tried: {climber.replicas[0]["perturbation_num"]}')
   print(f'Steps accepted: {climber.replicas[0]["num_accepted"]}')
   print(f'Improvements found: {climber.replicas[0]["num_improvements"]}')

Real-time monitoring
--------------------

You can monitor in-progress optimizations with the built-in Streamlit dashboard. To use the dashboard, install hill climber with the dashboard extras and then launch the dashboard:

.. code-block:: bash

   $ pip install parallel-hill-climber[dashboard]
   $ hill-climber-dashboard

     You can now view your Streamlit app in your browser.

     Local URL: http://localhost:8501
     Network URL: http://172.17.0.2:8501

Access the dashboard via the URL provided. Note: the dashboard is only available on the same machine (or same LAN) running hill climber.

Next steps
----------
Replica exchange (parallel tempering)
--------------------------------------

Hill Climber uses replica exchange (parallel tempering) by default. Multiple
replicas run at different temperatures and exchange configurations to improve
global optimization:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=objective_high_correlation,
       max_time=10,
       mode='maximize',
       n_replicas=8,           # Number of replicas (default: 4)
       T_min=0.0001,           # Minimum temperature (default: 0.0001)
       T_max=10.0,             # Maximum temperature (default: 100 * T_min)
       exchange_interval=100   # Steps between exchange attempts (default: 100)
   )

   best_data = climber.climb()

The ``climb()`` method automatically runs all replicas and returns the best result.