Installation
============

Requirements
------------

- Python 3.10 or higher
- NumPy
- Pandas
- Matplotlib
- SciPy
- Numba

Install from PyPI
-----------------

Install the package directly from PyPI to use it in your own projects:

.. code-block:: bash

   pip install parallel-hill-climber

This is the recommended method for using Hill Climber in your code.

Development Installation
------------------------

To explore the examples, modify the code, or contribute:

**Option 1: GitHub Codespaces (No local setup required)**

1. Fork the repository on GitHub
2. Open in GitHub Codespaces
3. The development environment will be configured automatically

Test that the installation was successful:

.. code-block:: python

   import hill_climber
   print(f"Hill Climber {hill_climber.__version__} successfully installed!")

**Option 2: Local Development**

1. Clone or fork the repository:

   .. code-block:: bash

      git clone https://github.com/gperdrizet/hill_climber.git
      cd hill_climber

2. Install in editable mode:

   .. code-block:: bash

      pip install -e .

3. Running Tests

After installation, verify everything works by running the test suite:

.. code-block:: bash

   python -m pytest tests/

All tests should pass.
