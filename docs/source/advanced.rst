Advanced Topics
===============

Custom Objective Functions
---------------------------

Complex Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hill Climber supports n-dimensional data. Your objective function should accept
as many arguments as you have columns. Combine multiple objectives with different weights:

.. code-block:: python

   def multi_objective(w, x, y, z):
       """Optimize multiple properties simultaneously for 4D data."""
       
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
- **CPU resource limit**: Not recommended to run more replicates than one less than avalible CPU threads

Replicate Noise Tuning
~~~~~~~~~~~~~~~~~~~~~~~

- **Low noise** (~1%-10% of input range): When starting data is already close to solutions
- **Medium noise** (~10%-50% of input range): General purpose exploration
- **High noise** (~50%-150% of input range): When you need very diverse starting points

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

   resumed = HillClimber.resume_from_checkpoint(
       checkpoint_file='optimization.pkl',
       objective_func=my_objective,
       new_max_time=30  # Continue for 30 more minutes
   )
   
   result = resumed.climb()

Progress Monitoring
-------------------

Live Progress Plots
~~~~~~~~~~~~~~~~~~~

Monitor optimization progress in real-time with automatic plotting:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       max_time=60,
       plot_progress=5  # Plot every 5 minutes
   )
   
   result = climber.climb()

This is particularly useful for:

- Long-running optimizations (>10 minutes)
- Interactive Jupyter notebooks
- Debugging objective functions
- Monitoring convergence behavior

**Important Notes**:

- Only works with ``climb()`` method (single-process mode)
- Does **not** work with ``climb_parallel()`` because worker processes don't
  report intermediate results
- If no steps are accepted between plot intervals, displays time information
  instead of plotting
- In Jupyter notebooks, each plot replaces the previous one for clean output

Performance Optimization
------------------------

Perturbation Strategies
~~~~~~~~~~~~~~~~~~~~~~~

**Perturbation distribution**:

Perturbations are sampled from a normal distribution N(0, ``step_spread``):

- Mean is always 0 (symmetric perturbations around current values)
- ``step_spread``: Standard deviation (controls magnitude variability)
- Default ``step_spread=1.0`` provides moderate-sized perturbations

Example:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       perturb_fraction=0.1,  # perturb 10% of elements
       step_spread=0.5        # moderate variability
   )

Faster Convergence
~~~~~~~~~~~~~~~~~~

For quick convergence, use aggressive parameters:

- **Large step_spread** (5.0-10.0): Allow bigger perturbations
- **High perturb_fraction** (0.4-0.6): Modify more points
- **Low temperature** (10-50): More greedy optimization
- **Slower cooling** (0.0001): More exploration of suboptimal solutions

Better Exploration
~~~~~~~~~~~~~~~~~~

For thorough exploration of solution space:

- **Small step_spread** (0.1-0.5): Precise adjustments
- **Low perturb_fraction** (0.1-0.2): Subtle changes
- **High temperature** (100-500): Accept more suboptimal moves
- **Faster cooling** (0.01-0.001): Gradual convergence

Algorithm Visualization
-----------------------

The hill climbing process can be visualized as searching a fitness landscape.
The algorithm:

1. Starts from initial data
2. Makes random perturbations sampled from a normal distribution N(0, ``step_spread``)
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

- Increase ``step_spread`` for larger perturbations
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

- Decrease ``step_spread`` for finer control
- Decrease ``temperature`` to be more selective
- Check for bugs in objective function
- Ensure objective weights are balanced

Package Information
-------------------

Version Information
~~~~~~~~~~~~~~~~~~~

To check the installed version:

.. code-block:: python

   import hill_climber
   print(hill_climber.__version__)

The package follows semantic versioning (MAJOR.MINOR.PATCH).

License
~~~~~~~

Hill Climber is licensed under the GNU General Public License v3.0 (GPL-3.0).
You are free to use, modify, and distribute this software, but any derivative
works must also be released under the GPL-3.0 license.

Citation
~~~~~~~~

If you use this package in your research, please cite it appropriately.
Visit the `GitHub repository <https://github.com/gperdrizet/hill_climber>`_
and click the "Cite this repository" button for properly formatted citations
in APA, BibTeX, or other formats.
