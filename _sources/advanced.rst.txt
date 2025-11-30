Advanced Topics
===============

Custom Objective Functions
---------------------------

Complex Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hill Climber supports multi-column data. Your objective function should accept
as many arguments as you have columns. Combine multiple objectives with different weights:

.. code-block:: python

   def multi_objective(w, x, y, z):
       """Optimize multiple properties simultaneously for 4-column data."""
       
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

Replica Exchange Tuning
------------------------

Temperature Ladder Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the appropriate temperature range and spacing:

.. code-block:: python

   # Wide temperature range for difficult landscapes
   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       n_replicas=8,
       T_min=0.1,            # Coldest replica
       T_max=100.0,          # Hottest replica  
       temperature_scheme='geometric'  # Recommended for better mixing
   )

   # Narrow range for fine-tuning
   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       n_replicas=4,
       T_min=1.0,
       T_max=5.0,
       temperature_scheme='linear'
   )

Exchange Strategy Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different strategies for replica pairing:

- **even_odd** (default): Alternates between even and odd pairs (0-1, 2-3, then 1-2, 3-4). Good balance of mixing and efficiency.
- **random**: Random pair selection each round. More stochastic exploration.
- **all_neighbors**: All neighboring pairs attempt exchange each round. More thorough but slower.

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       exchange_strategy='random',  # or 'even_odd', 'all_neighbors'
       exchange_interval=10000  # Exchange attempts every 10000 steps
   )

Choosing Number of Replicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **4 replicas**: Good default for most problems
- **8-12 replicas**: Better exploration of complex landscapes
- **16+ replicas**: For very difficult optimization problems
- **Memory consideration**: Each replica maintains a copy of your data

Trade-offs:
- More replicas = better exploration but more memory usage
- Fewer replicas = faster per-iteration but may miss global optima

Checkpointing
-------------

For long optimizations, save intermediate progress:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       max_time=60,
       checkpoint_file='long_run.pkl'  # Auto-saves after each batch
   )
   
   result = climber.climb()

Resume from a checkpoint:

.. code-block:: python

   # Continue with saved temperatures (default)
   resumed = HillClimber.load_checkpoint(
       filepath='optimization.pkl',
       objective_func=my_objective
   )
   
   # Or reset temperatures to restart cooling schedule
   resumed = HillClimber.load_checkpoint(
       filepath='optimization.pkl',
       objective_func=my_objective,
       reset_temperatures=True  # Restart from hot temperatures
   )
   
   # Continue from where it left off
   best_data, history_df = resumed.climb()

Temperature Reset on Resume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``load_checkpoint`` preserves the cooled temperatures from the saved state,
allowing the optimization to continue its cooling schedule. However, you can reset
temperatures to their original ladder values:

.. code-block:: python

   # Reset temperatures when resuming
   resumed = HillClimber.load_checkpoint(
       filepath='optimization.pkl',
       objective_func=my_objective,
       reset_temperatures=True
   )

**When to reset temperatures:**

- **Escaped local minimum**: If the optimization found a good solution but you want to explore more aggressively
- **Multiple restart strategy**: Run multiple sessions with fresh temperatures for better exploration
- **Stuck optimization**: Replicas have cooled too much and accept very few moves

**When to keep saved temperatures (default):**

- **Continuing long optimization**: Natural continuation of the cooling schedule
- **Refining solution**: Cooler temperatures help fine-tune the current best solution
- **Limited time remaining**: Use remaining time efficiently without re-exploring

Note that resetting temperatures restarts the cooling schedule but preserves all other
state including current configurations, best solutions, and optimization history.

Performance Optimization
------------------------

Perturbation Strategies
~~~~~~~~~~~~~~~~~~~~~~~

**Perturbation distribution**:

Perturbations are sampled from a normal distribution N(0, σ) where σ is calculated
as ``step_spread * mean(data_range)``:  

- Mean is always 0 (symmetric perturbations around current values)
- ``step_spread``: Fraction of data range (default: 0.01 = 1%)
- Actual standard deviation scales with your data automatically

Example:

.. code-block:: python

   climber = HillClimber(
       data=data,
       objective_func=my_objective,
       perturb_fraction=0.001,  # perturb 0.1% of elements (default)
       step_spread=0.02         # 2% of data range
   )Faster Convergence
~~~~~~~~~~~~~~~~~~

For quick convergence, use aggressive parameters:

- **Large step_spread** (0.05-0.10): Allow bigger perturbations (5-10% of range)
- **High perturb_fraction** (0.01-0.1): Modify more points
- **Low T_min** (0.01-0.1): More greedy optimization
- **Higher cooling_rate** (1e-6): Faster temperature reduction

Better Exploration
~~~~~~~~~~~~~~~~~~

For thorough exploration of solution space:

- **Small step_spread** (0.001-0.005): Precise adjustments (0.1-0.5% of range)
- **Low perturb_fraction** (0.0001-0.001): Subtle changes
- **High T_min** (1.0-10.0): Accept more suboptimal moves
- **Lower cooling_rate** (1e-9 to 1e-10): Gradual convergence

Algorithm Visualization
-----------------------

The hill climbing process can be visualized as searching a fitness landscape.
The algorithm:

1. Starts from initial data
2. Makes random perturbations sampled from N(0, ``step_spread * mean(data_range)``)
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

- Increase ``step_spread`` for larger perturbations (try 0.05-0.10)
- Increase ``perturb_fraction`` to modify more points
- Decrease ``T_min`` for more greedy optimization
- Check if objective function has bugs or is too constrained

Converging to Local Optima
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Different runs find similar suboptimal solutions, exchange acceptance rate is very low

**Solutions**:

- Increase ``T_max`` for hotter replicas to explore more broadly
- Increase ``n_replicas`` for better temperature coverage
- Use smaller ``cooling_rate`` (slower cooling) to explore longer
- Adjust ``exchange_interval`` (try smaller values for more frequent exchanges)
- Check temperature ladder - ensure good spacing between replicas

Oscillating Objective Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Objective improves then worsens repeatedly

**Solutions**:

- Decrease ``step_spread`` for finer control (try 0.005-0.01)
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
Visit the `GitHub repository <https://github.com/gperdrizet/hill_climber>`__ (opens in new tab)
and click the "Cite this repository" button for properly formatted citations
in APA, BibTeX, or other formats.

.. raw:: html

   <script>
   document.querySelectorAll('a[href="https://github.com/gperdrizet/hill_climber"]').forEach(function(link) {
       link.setAttribute('target', '_blank');
   });
   </script>
