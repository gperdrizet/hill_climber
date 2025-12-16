Example notebooks
=================

The following Jupyter notebooks demonstrate various applications of Hill Climber.

Notebook descriptions
---------------------

1. Simulated annealing introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Introduction to simulated annealing concepts and the hill climbing algorithm.

View: :download:`01-simulated_annealing.ipynb <../../notebooks/01-simulated_annealing.ipynb>`

2. Pearson & Spearman correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate datasets with:

- Strong Spearman correlation but weak Pearson correlation (non-linear monotonic)
- Strong Pearson correlation but weak Spearman correlation (linear with outliers)

View: :download:`02-pearson_spearman.ipynb <../../notebooks/02-pearson_spearman.ipynb>`

3. Mean & standard deviation with diverse structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create families of distributions with:

- Identical means across distributions
- Identical standard deviations across distributions
- Maximum structural diversity (different shapes)

View: :download:`03-mean_std.ipynb <../../notebooks/03-mean_std.ipynb>`

4. Low Pearson correlation & low entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate 2D point distributions with:

- Low Pearson correlation (near-zero linear relationship)
- Low joint entropy (clustered, non-uniform distributions)

View: :download:`04-entropy_pearson.ipynb <../../notebooks/04-entropy_pearson.ipynb>`

5. Feature interactions
~~~~~~~~~~~~~~~~~~~~~~~~

Create datasets where:

- Individual features have weak correlations with the label
- Multiple linear regression using all features achieves high R²
- Demonstrates importance of feature interactions

View: :download:`05-feature_interactions.ipynb <../../notebooks/05-feature_interactions.ipynb>`

6. Checkpointing example
~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates checkpoint and resume functionality for long-running optimizations.

View: :download:`06-checkpoint_example.ipynb <../../notebooks/06-checkpoint_example.ipynb>`

.. note::
   To run these notebooks interactively, clone the repository and open them in Jupyter:
   
   .. code-block:: bash
   
      git clone https://github.com/gperdrizet/hill_climber.git
      cd hill_climber
      jupyter notebook notebooks/
