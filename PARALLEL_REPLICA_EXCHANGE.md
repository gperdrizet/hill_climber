# Parallel Replica Exchange

## Overview

The hill_climber package now supports parallel execution of replica exchange using multiprocessing. This allows replicas to run simultaneously on different CPU cores, significantly improving performance for computationally intensive objective functions.

## Key Features

- **True parallelization**: Replicas run in separate processes on different CPU cores
- **Automatic worker management**: Worker pool is created and managed automatically
- **Flexible configuration**: Choose number of workers or fall back to sequential mode
- **Seamless integration**: Same API as sequential mode with one additional parameter
- **Checkpoint compatibility**: Parallel and sequential modes share the same checkpoint format

## Usage

### Basic Parallel Execution

```python
from hill_climber import HillClimber
import pandas as pd

# Create your data and objective function
data = pd.DataFrame({'x': ..., 'y': ...})

def objective_func(x, y):
    # Your objective function
    metrics = {...}
    objective_value = ...
    return metrics, objective_value

# Create climber with parallel execution
climber = HillClimber(
    data=data,
    objective_func=objective_func,
    n_replicas=4,
    n_workers=4  # Use 4 parallel workers
)

best_data, history = climber.climb()
```

### Worker Configuration Options

```python
# Use default: n_workers = n_replicas
climber = HillClimber(..., n_replicas=4)  # 4 workers

# Specify number of workers explicitly
climber = HillClimber(..., n_replicas=8, n_workers=4)  # 8 replicas, 4 workers

# Sequential mode (original behavior)
climber = HillClimber(..., n_replicas=4, n_workers=0)  # No parallelization

# Use all available CPU cores
import multiprocessing
climber = HillClimber(..., n_workers=multiprocessing.cpu_count())
```

## How It Works

### Architecture

1. **Main Process**: Coordinates optimization, handles replica exchanges, checkpointing, and plotting
2. **Worker Processes**: Execute optimization steps for assigned replicas independently
3. **Synchronization Points**: After each batch of steps (defined by `exchange_interval`), workers return results for exchange

### Execution Flow

```
Main Process:
  ├─ Initialize replicas with temperature ladder
  ├─ Create worker pool
  └─ For each exchange interval:
      ├─ Distribute replica states to workers (parallel)
      ├─ Workers execute n steps independently
      ├─ Collect updated states from workers (blocking)
      ├─ Perform replica exchanges (sequential)
      ├─ Save checkpoint if needed (sequential)
      └─ Plot progress if needed (sequential)
```

### Worker Process

Each worker:
1. Receives serialized replica state
2. Performs `exchange_interval` optimization steps
3. Returns updated state to main process
4. Repeats with new state after exchange

## Performance Considerations

### When to Use Parallel Mode

**Best for:**
- Computationally expensive objective functions
- Large datasets (> 1000 points)
- Complex simulations or calculations in objective function
- Multi-core systems with 4+ CPUs

**May not help:**
- Very fast objective functions (< 1ms)
- Small datasets (< 100 points)
- I/O-bound objective functions
- Single or dual-core systems

### Expected Speedup

Speedup depends on:
- **Objective function complexity**: More computation = better speedup
- **Number of cores**: More cores = higher potential speedup
- **Serialization overhead**: Large data arrays reduce efficiency
- **Exchange frequency**: Lower `exchange_interval` = more synchronization overhead

Typical speedups:
- Fast objective functions: 1.5-2x with 4 workers
- Medium complexity: 2-3x with 4 workers
- Heavy computations: 3-4x with 4 workers (near-linear scaling)

### Optimization Tips

1. **Increase exchange_interval**: Reduce synchronization overhead
   ```python
   climber = HillClimber(..., exchange_interval=500)  # Instead of 100
   ```

2. **Match workers to cores**: Don't exceed available CPUs
   ```python
   import multiprocessing
   n_workers = min(n_replicas, multiprocessing.cpu_count())
   ```

3. **Profile your objective function**: Use `%timeit` to measure evaluation time
   ```python
   %timeit objective_func(data['x'], data['y'])
   # If < 1ms, parallel mode may not help
   ```

4. **Monitor CPU usage**: Verify cores are actually being used
   ```bash
   htop  # or top on macOS
   ```

## Limitations

### Serialization Requirements

Objective functions must be:
- **Picklable**: No lambda functions or local closures
- **Self-contained**: All imports must be at module level
- **Pure functions**: Avoid global state

**Good:**
```python
from scipy.stats import pearsonr

def objective_func(x, y):
    corr, _ = pearsonr(x, y)
    return {'correlation': corr}, corr
```

**Bad:**
```python
# Lambda - not picklable
objective_func = lambda x, y: ({'corr': pearsonr(x, y)[0]}, pearsonr(x, y)[0])

# Uses external variable
threshold = 0.5
def objective_func(x, y):
    corr = pearsonr(x, y)[0]
    return {'corr': corr}, corr if corr > threshold else 0
```

### Memory Considerations

- Each worker holds a copy of the data
- Peak memory ≈ (n_workers + 1) × data_size
- Large datasets (> 1GB) may cause memory issues with many workers

## Troubleshooting

### "Can't pickle function"

**Problem**: Objective function cannot be serialized

**Solution**: Define function at module level, not inline
```python
# Put this in a separate .py file or at top level
def my_objective(x, y):
    return metrics, value

# Then import and use
from my_module import my_objective
climber = HillClimber(objective_func=my_objective, ...)
```

### Slower than sequential mode

**Problem**: Parallel mode is slower than expected

**Solutions**:
1. Increase `exchange_interval` to reduce overhead
2. Reduce number of workers if objective is fast
3. Profile objective function - may be too simple to benefit
4. Check if objective is I/O bound (file/network access)

### High memory usage

**Problem**: Running out of memory with parallel mode

**Solutions**:
1. Reduce `n_workers`
2. Reduce `n_replicas`
3. Use smaller dataset if possible
4. Switch to sequential mode (`n_workers=0`)

## Examples

See `test_parallel.py` and `benchmark_parallel.py` for working examples.

## Technical Details

### Communication

- Uses `multiprocessing.Pool` for worker management
- `Pool.map()` for distributing work
- `functools.partial` for parameter binding
- Pickle for state serialization

### Thread Safety

- Each worker has independent random state
- No shared memory between workers
- All exchanges happen in main process after worker completion
- Checkpoints saved only from main process

## Backward Compatibility

The parallel implementation is fully backward compatible:

- Default behavior unchanged (uses parallel if n_workers not specified)
- `n_workers=0` gives exact original sequential behavior
- Checkpoints work across both modes
- All existing code continues to work
