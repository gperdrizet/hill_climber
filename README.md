![Hill Climber](hill_climber/assets/readme-banner.svg)

[![PyPI Package](https://github.com/gperdrizet/hill_climber/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/gperdrizet/hill_climber/actions/workflows/publish-to-pypi.yml) [![Documentation](https://github.com/gperdrizet/hill_climber/actions/workflows/docs.yml/badge.svg)](https://github.com/gperdrizet/hill_climber/actions/workflows/docs.yml) [![PR Validation](https://github.com/gperdrizet/hill_climber/actions/workflows/pr-validation.yml/badge.svg)](https://github.com/gperdrizet/hill_climber/actions/workflows/pr-validation.yml)

A Python package for [hill climbing](https://pubsonline.informs.org/doi/10.1287/opre.21.2.498) optimization of user-supplied objective functions with [simulated annealing](https://cs.gettysburg.edu/~tneller/cs371/OptBySimAnneal.pdf) and [replica exchange](https://www.researchgate.net/publication/13255490_Replica_Monte_Carlo_Simulation_of_Spin-Glasses). Designed for flexible multi-objective optimization with support for multi-column datasets.

## 1. Documentation

**<a href="https://gperdrizet.github.io/hill_climber" target="_blank">Documentation on GitHub Pages</a>**

## 2. Features

- **Flexible objectives**: Support for user supplied objective functions with custom metrics
- **Real-time monitoring dashboard**: Live progress plots and run info. with SQLite backend
- **Replica exchange (parallel tempering)**: Replicas at different temperatures exchange configurations for improved global optimization (`multiprocessing.Pool`)
- **Simulated annealing**: Temperature-based acceptance of suboptimal solutions to escape local minima
- **Checkpoint/resume**: Save and resume long-running optimizations with configurable checkpoint intervals
- **JIT compilation**: Numba-optimized core functions for performance

## 3. Quick start

### 3.1. Installation

Install the package directly from PyPI to use it in your own projects:

```bash
pip install parallel-hill-climber
```

For detailed usage, configuration options, and advanced features, see the <a href="https://gperdrizet.github.io/hill_climber" target="_blank">full documentation</a>.

### 3.2. Example climb

Simple hill climb to maximize the Pearson correlation coefficient between two random uniform features:

```python
import numpy as np
import pandas as pd

from hill_climber import HillClimber

# Create sample data
data = pd.DataFrame({
    'x': np.random.rand(100),
    'y': np.random.rand(100)
})

# Define objective function
def my_objective(x, y):

    correlation = pd.Series(x).corr(pd.Series(y))
    metrics = {'correlation': correlation}

    return metrics, correlation

# Create hill climber instance
climber = HillClimber(
    data=data,
    objective_func=my_objective,
    max_time=1,
    mode='maximize',
    n_replicas=4
)

# Run optimization
best_data = climber.climb()
```

The return `best_data` contains the winning solution from all replicates at the end of the run. Individual replicate results can be accessed with the climber object's `.get_replicas()` method after the run is complete.

### 3.3. Real-time monitoring dashboard

You can monitor in-progress optimizations with the built-in Streamlit dashboard. To use the dashboard, install hill climber with the dashboard extras and then launch the dashboard:

```text
$ pip install parallel-hill-climber[dashboard]
$ hill-climber-dashboard

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.17.0.2:8501
```

Access the dashboard via the URL provided. Note: the dashboard is only available on the same machine (or same LAN) running hill climber.

![Dashboard Screenshot](docs/source/dashboard.png)


## 4. Development environment setup

To explore the examples, modify the code, or contribute:

### 4.1. Setup option 1: GitHub Codespaces (No local setup required)

1. Fork this repository
2. Open in GitHub Codespaces
3. The development environment will be configured automatically
4. Documentation will be built and served at http://localhost:8000 automatically
5. The monitoring dashboard will start at http://localhost:8501 automatically

### 4.2. Setup option 2: Local development

1. Clone or fork the repository:
   ```bash
   git clone https://github.com/gperdrizet/hill_climber.git
   cd hill_climber
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 4.3. Building documentation

You can build and view a local copy of the documentation as follows:

```bash
cd docs
make html
```

View docs by opening docs/build/html/index.html in a browser, or serve locally with: `python -m http.server 8000 --directory build/html`

### 4.4. Running tests

To run the test suite:

```bash
# Run all tests
python tests/run_tests.py

# Or with pytest if installed
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_hill_climber.py
```

## 5. Contributing

Contributions welcome! Please ensure all tests pass before submitting pull requests.

## 6. License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for full details.

In summary, you are free to use, modify, and distribute this software, but any derivative works must also be released under the GPL-3.0 license.

And please cite!

## 7. Citation

If you use this package in your research, please use the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/gperdrizet/hill_climber) to get properly formatted citations in APA, BibTeX, or other formats.
