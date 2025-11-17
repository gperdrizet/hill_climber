# Hill Climber

[![PyPI Package](https://github.com/gperdrizet/hill_climber/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/gperdrizet/hill_climber/actions/workflows/publish-to-pypi.yml) [![Documentation](https://github.com/gperdrizet/hill_climber/actions/workflows/docs.yml/badge.svg)](https://github.com/gperdrizet/hill_climber/actions/workflows/docs.yml) [![PR Validation](https://github.com/gperdrizet/hill_climber/actions/workflows/pr-validation.yml/badge.svg)](https://github.com/gperdrizet/hill_climber/actions/workflows/pr-validation.yml)

A Python package for hill climbing optimization of user-supplied objective functions with simulated annealing. Designed for flexible multi-objective optimization with support for N-dimensional data.

## 1. Documentation

**<a href="https://gperdrizet.github.io/hill_climber" target="_blank">View Full Documentation on GitHub Pages</a>**

## 2. Features

- **Simulated Annealing**: Temperature-based acceptance of suboptimal solutions to escape local minima
- **Parallel Execution**: Run multiple replicates simultaneously for diverse solutions
- **Flexible Objectives**: Support for any objective function with multiple metrics
- **N-Dimensional Support**: Optimize distributions with any number of dimensions
- **Checkpoint/Resume**: Save and resume long-running optimizations
- **Boundary Handling**: Reflection-based strategy prevents point accumulation at boundaries
- **Visualization**: Built-in plotting for both input data and optimization results
- **JIT Compilation**: Numba-optimized core functions for performance

## 3. Quick Start

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

# Create optimizer
climber = HillClimber(
    data=data,
    objective_func=my_objective,
    max_time=1,  # minutes
    mode='maximize'
)

# Run optimization with multiple replicates
results = climber.climb_parallel(replicates=4, initial_noise=0.1)

# Visualize results
climber.plot_results(results, plot_type='histogram')
```

### 3.3. Example Notebooks

The `notebooks/` directory contains demonstration of key concepts and complete worked examples demonstrating various use cases:

1. **<a href="https://github.com/gperdrizet/hill_climber/blob/main/notebooks/01-simulated_annealing.ipynb" target="_blank">Simulated Annealing</a>**: Introduction to simulated annealing algorithm
2. **<a href="https://github.com/gperdrizet/hill_climber/blob/main/notebooks/02-pearson_spearman.ipynb" target="_blank">Pearson & Spearman</a>**: Optimizing for different correlation measures
3. **<a href="https://github.com/gperdrizet/hill_climber/blob/main/notebooks/03-mean_std.ipynb" target="_blank">Mean & Std</a>**: Creating distributions with matching statistics but diverse structures
4. **<a href="https://github.com/gperdrizet/hill_climber/blob/main/notebooks/04-entropy_pearson.ipynb" target="_blank">Entropy & Correlation</a>**: Low correlation with internal structure
5. **<a href="https://github.com/gperdrizet/hill_climber/blob/main/notebooks/05-feature_interactions.ipynb" target="_blank">Feature Interactions</a>**: Machine learning feature engineering demonstrations
6. **<a href="https://github.com/gperdrizet/hill_climber/blob/main/notebooks/06-checkpoint_example.ipynb" target="_blank">Checkpointing</a>**: Long-running optimization with save/resume


## 4. Development Environment Setup

To explore the examples, modify the code, or contribute:

### 4.1. Setup Option 1: GitHub Codespaces (No local setup required)

1. Fork this repository
2. Open in GitHub Codespaces
3. The development environment will be configured automatically
4. Documentation will be built and served at http://localhost:8000 automatically

### 4.2. Setup Option 2: Local Development

1. Clone or fork the repository:
   ```bash
   git clone https://github.com/gperdrizet/hill_climber.git
   cd hill_climber
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 4.3. Building Documentation

You can build and view a local copy of the documentation as follows:

```bash
cd docs
make html
# View docs by opening docs/build/html/index.html in a browser
# Or serve locally with: python -m http.server 8000 --directory build/html
```

### 4.4. Running Tests

To run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_hill_climber.py

# Run with coverage
python -m pytest tests/ --cov=hill_climber
```

## 5. Contributing

Contributions welcome! Please ensure all tests pass before submitting pull requests.

## 6. License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for full details.

In summary, you are free to use, modify, and distribute this software, but any derivative works must also be released under the GPL-3.0 license.

## 7. Citation

If you use this package in your research, please use the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/gperdrizet/hill_climber) to get properly formatted citations in APA, BibTeX, or other formats.
