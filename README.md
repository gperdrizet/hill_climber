# Hill Climber

A Python package for hill climbing optimization of user-supplied objective functions with simulated annealing. Designed for flexible multi-objective optimization with support for N-dimensional data.

## Documentation

**[View Full Documentation on GitHub Pages](https://gperdrizet.github.io/hill_climber/)**

## Features

- **Simulated Annealing**: Temperature-based acceptance of suboptimal solutions to escape local minima
- **Parallel Execution**: Run multiple replicates simultaneously for diverse solutions
- **Flexible Objectives**: Support for any objective function with multiple metrics
- **N-Dimensional Support**: Optimize distributions with any number of dimensions
- **Checkpoint/Resume**: Save and resume long-running optimizations
- **Boundary Handling**: Reflection-based strategy prevents point accumulation at boundaries
- **Visualization**: Built-in plotting for both input data and optimization results
- **JIT Compilation**: Numba-optimized core functions for performance

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Matplotlib
- Numba

## Quick Start

```python
from hill_climber import HillClimber
import pandas as pd
import numpy as np

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
    step_size=0.5,
    mode='maximize'
)

# Run optimization with multiple replicates
results = climber.climb_parallel(replicates=4, initial_noise=0.1)

# Visualize results
climber.plot_results(results, plot_type='histogram')
```

For detailed usage, configuration options, and advanced features, see the [full documentation](https://gperdrizet.github.io/hill_climber/).

## Example Notebooks

The `notebooks/` directory contains complete worked examples demonstrating various use cases:

1. **Simulated Annealing**: Introduction to the algorithm
2. **Pearson & Spearman**: Optimizing for different correlation measures
3. **Mean & Std**: Creating distributions with matching statistics but diverse structures
4. **Entropy & Correlation**: Low correlation with internal structure
5. **Feature Interactions**: Machine learning feature engineering demonstrations
6. **Checkpointing**: Long-running optimization with save/resume

See the [documentation](https://gperdrizet.github.io/hill_climber/notebooks.html) for rendered versions of all notebooks.

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_hill_climber.py

# Run with coverage
python -m pytest tests/ --cov=hill_climber
```

All 53 tests passing ✓

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure all tests pass before submitting pull requests.

## Citation

If you use this package in your research, please cite appropriately.
