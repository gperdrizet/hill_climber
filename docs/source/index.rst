Hill Climber Documentation
==========================

A flexible simulated annealing optimizer for generating synthetic datasets with specific statistical properties.

**Hill Climber** is a Python package that uses simulated annealing to optimize datasets according to user-defined objective functions. It's particularly useful for:

- Generating synthetic data with specific statistical properties
- Exploring relationships between different correlation measures
- Creating datasets for testing and benchmarking
- Educational demonstrations of optimization algorithms

Installation
------------

.. code-block:: bash

   pip install parallel-hill-climber

See :doc:`installation` for more options including development setup.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api
   notebooks
   advanced

Features
--------

- **Flexible objective functions**: Define custom objectives for any statistical property
- **Simulated annealing**: Escape local optima and find global solutions
- **Parallel replicates**: Run multiple optimizations simultaneously
- **Checkpointing**: Save and resume long-running optimizations
- **Rich visualization**: Built-in plotting for results analysis
- **2D data optimization**: Optimize datasets with two variables (x, y)
- **JIT Compilation**: Numba-optimized core functions for performance
- **Flexible perturbation**: Element-wise or row-wise perturbation strategies

Python Version Support
---------------------

- Python 3.10+
- Tested on Python 3.10, 3.11, and 3.12

Links
-----

- **PyPI Package**: https://pypi.org/project/parallel-hill-climber/
- **GitHub Repository**: https://github.com/gperdrizet/hill_climber
- **Issue Tracker**: https://github.com/gperdrizet/hill_climber/issues

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
