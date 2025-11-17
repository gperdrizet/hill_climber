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

Option 1: GitHub Codespaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No local setup required:

1. Fork the repository on GitHub
2. Open in GitHub Codespaces
3. The development environment will be configured automatically
4. Documentation will be built and served at http://localhost:8000 automatically

Option 2: Local Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone and install locally:

1. Clone or fork the repository:

   .. code-block:: bash

      git clone https://github.com/gperdrizet/hill_climber.git
      cd hill_climber

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

4. Build documentation (optional):

   .. code-block:: bash

      cd docs
      make html
      # View docs by opening docs/build/html/index.html in a browser
      # Or serve locally with: python -m http.server 8000 --directory build/html

5. Run tests to verify installation:

   .. code-block:: bash

      # Run all tests
      python -m pytest tests/

      # Run specific test file
      python -m pytest tests/test_hill_climber.py

      # Run with coverage
      python -m pytest tests/ --cov=hill_climber

   All tests should pass.

Verifying Installation
^^^^^^^^^^^^^^^^^^^^^^

Test that the installation was successful:

.. code-block:: python

   import hill_climber
   print(f"Hill Climber {hill_climber.__version__} successfully installed!")
