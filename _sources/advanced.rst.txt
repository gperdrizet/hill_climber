Advanced Topics
===============

Custom Objective Functions
---------------------------

Complex Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine multiple objectives with different weights:

.. code-block:: python

   def multi_objective(w, x, y, z):
       """Optimize multiple properties simultaneously."""
       
       # Calculate individual objectives
       mean_similarity = calculate_mean_penalty(w, x, y, z)
       std_similarity = calculate_std_penalty(w, x, y, z)
       structural_diversity = calculate_ks_statistics(w, x, y, z)
       
       # Combine with weights
       objective = (
           10.0 * structural_diversity -
           5.0 * mean_similarity -
           5.0 * std_similarity
       )
       
       metrics = {
           'Mean Similarity': mean_similarity,
           'Std Similarity': std_similarity,
           'Structural Diversity': structural_diversity
       }
       
       return metrics, objective

Handling Constraints
~~~~~~~~~~~~~~~~~~~~

Implement hard constraints through penalties:

.. code-block:: python

   def constrained_objective(x, y):
       """Optimize with constraints."""
       
       # Calculate main objective
       correlation = pearsonr(x, y)[0]
       
       # Check constraints
       penalty = 0.0
       
       # Constraint: mean must be near 0.5
       mean_x = np.mean(x)
       if abs(mean_x - 0.5) > 0.1:
           penalty += 100 * abs(mean_x - 0.5)
       
       # Constraint: std must be > 0.2
       std_x = np.std(x)
       if std_x < 0.2:
           penalty += 100 * (0.2 - std_x)
       
       objective = correlation - penalty
       
       return {'Correlation': correlation, 'Penalty': penalty}, objective

Parallel Processing Tips
-------------------------

Choosing Number of Replicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **4-8 replicates**: Good balance for most problems
- **More replicates**: Better exploration, find more diverse solutions
- **Fewer replicates**: Faster completion, use when solutions are similar

Replicate Noise Tuning
~~~~~~~~~~~~~~~~~~~~~~~

- **Low noise (0.1-0.3)**: When starting data is already close to solutions
- **Medium noise (0.3-0.7)**: General purpose exploration
- **High noise (0.7-1.5)**: When you need very diverse starting points

Checkpoint Strategies
---------------------

For Very Long Runs
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save every 10 minutes for 24-hour runs
   climber = HillClimber(
       data=data,
       objective_func=objective,
       max_time=1440,  # 24 hours
       checkpoint_file='long_run.pkl',
       save_interval=600  # 10 minutes
   )

Performance Optimization
------------------------

Faster Convergence
~~~~~~~~~~~~~~~~~~

For quick convergence, use aggressive parameters:

- **Large step_size** (2.0-5.0): Make bigger changes
- **High perturb_fraction** (0.4-0.6): Modify more points
- **Low temperature** (10-50): More greedy optimization
- **Slower cooling** (0.9999): More iterations

Better Exploration
~~~~~~~~~~~~~~~~~~

For thorough exploration of solution space:

- **Small step_size** (0.1-1.0): Precise adjustments
- **Low perturb_fraction** (0.1-0.2): Subtle changes
- **High temperature** (100-500): Accept more suboptimal moves
- **Faster cooling** (0.999-0.9995): Gradual convergence

Algorithm Visualization
-----------------------

The hill climbing process can be visualized as searching a fitness landscape.
The algorithm:

1. Starts from initial data
2. Makes random perturbations
3. Evaluates fitness via objective function
4. Accepts improvements (always) or worsening moves (with probability based on temperature)
5. Gradually reduces temperature to focus on local optimization
6. Returns the best solution found

Troubleshooting
---------------

No Progress After Many Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Objective value not improving, same metrics every iteration

**Solutions**:

- Increase ``step_size`` for larger perturbations
- Increase ``perturb_fraction`` to modify more points
- Decrease ``temperature`` for more greedy optimization
- Check if objective function has bugs or is too constrained

Converging to Local Optima
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Different replicates find similar suboptimal solutions

**Solutions**:

- Increase ``temperature`` for more exploration
- Increase ``initial_noise`` for more diverse starting points
- Use smaller ``cooling_rate`` (slower cooling) to explore longer
- Increase number of replicates

Oscillating Objective Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Objective improves then worsens repeatedly

**Solutions**:

- Decrease ``step_size`` for finer control
- Decrease ``temperature`` to be more selective
- Check for bugs in objective function
- Ensure objective weights are balanced
