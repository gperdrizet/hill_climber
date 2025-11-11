# Hill Climber

A Python package for hill climbing optimization of user-supplied objective function with simulated annealing. Designed for flexible multi-objective optimization with support for N-dimensional data.

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

## Key Concepts

### Optimization Modes

- **maximize**: Find maximum objective value
- **minimize**: Find minimum objective value  
- **target**: Approach a specific target value

### Replicate Noise

When running parallel optimization with multiple replicates, you can add uniform noise to create diverse starting points:

```python
results = climber.climb_parallel(replicates=8, initial_noise=2.0)
```

- **initial_noise**: Controls the magnitude of noise added to each replicate's starting data
- Noise is sampled uniformly: `±initial_noise`
- Each replicate gets different random noise, creating diverse exploration paths
- After noise addition, values are reflected back into bounds (see Boundary Handling)
- This helps find multiple diverse solutions rather than converging to the same local optimum

**Benefits:**
- Explores different regions of the solution space
- Increases chance of finding global optimum
- Generates diverse solutions for comparison

### Boundary Handling

Optimization is constrained to the input range. The optimizer uses a **reflection strategy** instead of clipping to handle values that exceed bounds:
- Values that go below minimum are reflected back: `new_value = min + (min - value)`
- Values that go above maximum are reflected back: `new_value = max - (value - max)`
- This prevents accumulation of points at boundaries (~77x improvement over clipping)
- Applies to both initial noise and optimization perturbations

### Result Structure

`climb_parallel()` returns a dictionary:
```python
{
    'input_data': <original data before noise>,
    'results': [
        (noisy_initial_1, best_data_1, steps_df_1),
        (noisy_initial_2, best_data_2, steps_df_2),
        ...
    ]
}
```

- `input_data`: Original input data (saved once)
- `noisy_initial`: Starting data for each replicate (after noise addition)
- `best_data`: Best solution found by that replicate
- `steps_df`: DataFrame tracking optimization progress with all metrics

## Example Notebooks

### 1. Simulated Annealing (`01-simulated_annealing.ipynb`)
Introduction to simulated annealing concepts and how the algorithm works.

### 2. Pearson & Spearman Correlation (`02-pearson_spearman.ipynb`)
Optimize for distributions where Pearson and Spearman correlations differ significantly.

### 3. Mean & Std with Diverse Structures (`03-mean_std.ipynb`)
Generate 4 distributions with:
- Same mean and standard deviation
- Maximum structural diversity (different shapes)
- Demonstrates N-dimensional optimization

### 4. Low Pearson Correlation & Low Entropy (`04-entropy_pearson.ipynb`)
Create low-correlation, low-entropy 2D distributions with internal structure.

### 5. Checkpoint Example (`05-checkpoint_example.ipynb`)
Demonstrates checkpoint/resume functionality for long-running optimizations.

## Advanced Features

### Checkpointing

```python
# Save checkpoints during optimization
climber = HillClimber(
    data=data,
    objective_func=my_objective,
    max_time=10,
    checkpoint_file='checkpoint.pkl',
    save_interval=60  # Save every 60 seconds
)

# Resume from checkpoint
climber_resumed = HillClimber.resume_from_checkpoint(
    checkpoint_file='checkpoint.pkl',
    objective_func=my_objective,
    new_max_time=10  # Additional time
)
results = climber_resumed.climb()
```

### Custom Objective Functions

Objective functions should:
1. Accept one argument per data column
2. Return `(metrics_dict, objective_value)`

```python
def objective_diverse_structures(w, x, y, z):
    from scipy import stats
    from itertools import combinations
    
    # Calculate statistics
    means = {
        'w': np.mean(w), 'x': np.mean(x),
        'y': np.mean(y), 'z': np.mean(z)
    }
    
    # Measure diversity with KS statistics
    distributions = {'w': w, 'x': x, 'y': y, 'z': z}
    ks_values = []
    
    for name1, name2 in combinations(distributions.keys(), 2):

        ks_stat, _ = stats.ks_2samp(
            distributions[name1], 
            distributions[name2]
        )
        ks_values.append(ks_stat)
    
    # Calculate objective
    mean_ks = np.mean(ks_values)
    target_mean = 10.0

    mean_penalty = np.mean([
        abs(m - target_mean) for m in means.values()
    ])
    
    objective = mean_ks - 2.0 * mean_penalty
    
    metrics = {
        **means,
        'mean_ks': mean_ks,
        'mean_penalty': mean_penalty
    }
    
    return metrics, objective
```

### Visualization Options

```python
# Plot input data
climber.plot_input(plot_type='scatter')  # or 'kde'

# Plot results with different styles
climber.plot_results(results, plot_type='scatter')
climber.plot_results(results, plot_type='histogram')

# Select specific metrics to display
climber.plot_results(
    results, 
    plot_type='histogram',
    metrics=['Mean Penalty', 'Mean KS Statistic']
)
```

## Hyperparameters

- **max_time**: Maximum optimization time in minutes
- **step_size**: Standard deviation of perturbations
- **perturb_fraction**: Fraction of data points to perturb per iteration (0.0-1.0)
- **temperature**: Initial temperature for simulated annealing
- **cooling_rate**: Temperature decay rate per iteration (0.0-1.0)
- **initial_noise**: Noise added to create diverse starting points for replicates

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

## Architecture

### Core Components

- **`optimizer.py`**: Main `HillClimber` class with optimization logic
- **`climber_functions.py`**: JIT-compiled helper functions (perturbation, objective calculation)
- **`plotting_functions.py`**: Visualization utilities

### Performance Optimizations

- Numba JIT compilation for core perturbation logic
- Efficient numpy operations
- Parallel execution with multiprocessing
- Reflection-based boundary handling (no clipping overhead)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure all tests pass before submitting pull requests.

## Citation

If you use this package in your research, please cite appropriately.
