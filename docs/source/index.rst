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
- **N-dimensional data**: Optimize datasets with any number of variables (x, y, z, ...)
- **Parallel replicates**: Run multiple optimizations simultaneously
- **Unified state management**: Clean dataclass architecture for internal state tracking
- **Checkpointing**: Save and resume long-running optimizations
- **Progress monitoring**: Live plotting during optimization runs
- **Rich visualization**: Built-in plotting for results analysis
- **JIT Compilation**: Numba-optimized core functions for performance

Python Version Support
----------------------

- Python 3.10+
- Tested on Python 3.10, 3.11, and 3.12

Links
-----

- **PyPI Package**: `parallel-hill-climber <https://pypi.org/project/parallel-hill-climber/>`__
- **GitHub Repository**: `hill_climber <https://github.com/gperdrizet/hill_climber>`__
- **Issue Tracker**: `Issues <https://github.com/gperdrizet/hill_climber/issues>`__

.. raw:: html

   <script>
   document.querySelectorAll('a[href*="github.com"], a[href*="pypi.org"]').forEach(function(link) {
       link.setAttribute('target', '_blank');
   });
   </script>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
