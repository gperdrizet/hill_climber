Installation
============

Requirements
------------

- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- SciPy
- Numba

Basic Installation
------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/gperdrizet/hill_climber.git
   cd hill_climber
   pip install -r requirements.txt

Verify Installation
-------------------

Test that the installation was successful:

.. code-block:: python

   from hill_climber import HillClimber
   print("Hill Climber successfully installed!")

Running Tests
-------------

After installation, verify everything works by running the test suite:

.. code-block:: bash

   pytest tests/

All tests should pass.
