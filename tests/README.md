# Hill Climber Unit Tests

This directory contains comprehensive unit tests for the hill climber package using Python's `unittest` framework.

## Test Files

### `test_climber_functions.py`
Tests for helper functions in `climber_functions.py`:
- **TestPerturbVectors**: Tests for the `perturb_vectors()` function
  - Array and DataFrame handling
  - Shape and column preservation
  - Positive value constraints
  - Data immutability
  
- **TestExtractColumns**: Tests for the `extract_columns()` function
  - Extraction from arrays and DataFrames
  - Column name preservation
  - Correct data types
  
- **TestCalculateObjective**: Tests for `calculate_objective()`
  - Array and DataFrame compatibility
  - Metrics dictionary structure
  - Integration with objective functions

### `test_hill_climber.py`
Tests for the `HillClimber` class:
- **TestHillClimberInitialization**: Tests for class initialization
  - Default and custom parameters
  - Mode validation (maximize, minimize, target)
  - Error handling for invalid inputs
  
- **TestHillClimberPrivateMethods**: Tests for internal methods
  - `_is_improvement()` for all modes
  - `_calculate_delta()` for all modes
  - `_record_improvement()` functionality
  
- **TestHillClimberClimb**: Tests for the `climb()` method
  - Return value structure
  - Data format preservation
  - Steps DataFrame creation
  - Multi-mode optimization
  
- **TestHillClimberClimbParallel**: Tests for parallel execution
  - Result structure validation
  - File saving functionality
  - CPU count validation
  - Initial noise parameter
  
- **TestHillClimberPlottingMethods**: Tests for plotting methods
  - Method existence and callability

## Running the Tests

### Run all tests:
```bash
python tests/run_tests.py
```

### Run specific test file:
```bash
python -m unittest tests/test_climber_functions.py
python -m unittest tests/test_hill_climber.py
```

### Run specific test class:
```bash
python -m unittest tests.test_climber_functions.TestPerturbVectors
python -m unittest tests.test_hill_climber.TestHillClimberInitialization
```

### Run specific test method:
```bash
python -m unittest tests.test_climber_functions.TestPerturbVectors.test_perturb_array_returns_array
```

### Run with verbose output:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Test Coverage

The test suite covers:
- **39 total tests**
- All helper functions
- HillClimber initialization and validation
- All optimization modes (maximize, minimize, target)
- Private methods and internal logic
- Single and parallel execution
- File I/O operations
- Data format preservation (numpy arrays and pandas DataFrames)
- Error handling and edge cases

## Test Design Principles

1. **Isolation**: Each test is independent and uses `setUp()` for fixtures
2. **Determinism**: Uses `np.random.seed()` for reproducible results
3. **Clarity**: Descriptive test names and docstrings
4. **Coverage**: Tests both success and failure paths
5. **Efficiency**: Short runtime (< 10 seconds for full suite)

## Notes

- Tests use module-level objective functions for multiprocessing compatibility
- Temporary files are automatically cleaned up after tests
- Random seeds are set for reproducible results
- Tests validate both functionality and error handling
