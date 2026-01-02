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

Development installation
------------------------

To explore the examples, modify the code, or contribute:

Option 1: GitHub Codespaces (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No local setup required. The dev container provides a complete, pre-configured environment:

**What's included:**

- Python 3.12 with all dependencies pre-installed
- Package installed in editable mode (code changes take effect immediately)
- Documentation server running at http://localhost:8000
- Real-time monitoring dashboard at http://localhost:8501
- Git LFS configured for large file handling
- Automatic cleanup of LFS cache to prevent disk space issues
- VS Code extensions for Python, Jupyter, and GitHub Actions

**Getting started:**

1. Fork the repository on GitHub
2. Click "Code" → "Codespaces" → "Create codespace on main"
3. Wait for the container to build (~2-3 minutes on first launch)
4. The environment is ready when you see the terminal prompt

**Available services:**

- **Documentation**: http://localhost:8000 - Auto-rebuilt on container attach
- **Dashboard**: http://localhost:8501 - Monitor optimization runs in real-time
- **Logs**: Dashboard logs available at ``/tmp/dashboard.log``

Option 2: Local development
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

4. Install the package in editable mode (required for dashboard CLI and imports):

   .. code-block:: bash

      pip install -e .

5. Launch the monitoring dashboard (optional):

   .. code-block:: bash

      hill-climber-dashboard

   Access at http://localhost:8501

6. Build documentation (optional):

   .. code-block:: bash

      cd docs
      make html

   View docs by opening ``docs/build/html/index.html`` in a browser, or serve locally:
   
   .. code-block:: bash

      python -m http.server 8000 --directory docs/build/html

7. Run tests to verify installation:

   .. code-block:: bash

      # Run all tests
      python -m pytest tests/

      # Run specific test file
      python -m pytest tests/test_hill_climber.py

   All tests should pass.

Verifying installation
^^^^^^^^^^^^^^^^^^^^^^

Test that the installation was successful:

.. code-block:: python

   import hill_climber
   print(f"Hill Climber {hill_climber.__version__} successfully installed!")
