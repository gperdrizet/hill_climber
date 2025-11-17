"""Unit tests for HillClimber class."""

import unittest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import pickle
from multiprocessing import cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hill_climber import HillClimber


def _simple_objective_for_parallel(x, y):
    """Simple objective function for parallel tests."""
    mean_val = np.mean(x) + np.mean(y)
    return {'mean': mean_val}, mean_val


class TestHillClimberInitialization(unittest.TestCase):
    """Test cases for HillClimber initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 4.0, 6.0]})
        
        def simple_objective(x, y):
            return {'mean': np.mean(x)}, np.mean(x)
        
        self.objective_func = simple_objective
    
    def test_initialization_with_defaults(self):
        """Test HillClimber initialization with default parameters."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func
        )
        self.assertEqual(climber.max_time, 30)
        self.assertEqual(climber.step_spread, 1.0)
        self.assertEqual(climber.temperature, 1000)
        self.assertEqual(climber.cooling_rate, 1 - 0.000001)
        self.assertEqual(climber.mode, 'maximize')
        self.assertIsNone(climber.target_value)
    
    def test_initialization_with_custom_params(self):
        """Test HillClimber initialization with custom parameters."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            max_time=5,
            step_spread=0.2,
            temperature=10.0,
            cooling_rate=0.01,
            mode='minimize'
        )
        self.assertEqual(climber.max_time, 5)
        self.assertEqual(climber.step_spread, 0.2)
        self.assertEqual(climber.temperature, 10.0)
        self.assertEqual(climber.cooling_rate, 1 - 0.01)
        self.assertEqual(climber.mode, 'minimize')
    
    def test_initialization_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            HillClimber(
                data=self.data,
                objective_func=self.objective_func,
                mode='invalid_mode'
            )
        self.assertIn("Mode must be", str(context.exception))
    
    def test_initialization_target_mode_without_value(self):
        """Test that target mode without target_value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            HillClimber(
                data=self.data,
                objective_func=self.objective_func,
                mode='target'
            )
        self.assertIn("target_value must be specified", str(context.exception))
    
    def test_initialization_target_mode_with_value(self):
        """Test that target mode with target_value works correctly."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='target',
            target_value=5.0
        )
        self.assertEqual(climber.mode, 'target')
        self.assertEqual(climber.target_value, 5.0)
    
    def test_initialization_with_array_data(self):
        """Test initialization with numpy array data."""
        array_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        climber = HillClimber(
            data=array_data,
            objective_func=self.objective_func
        )
        self.assertIsInstance(climber.data, np.ndarray)


class TestHillClimberPrivateMethods(unittest.TestCase):
    """Test cases for HillClimber private methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 4.0, 6.0]})
        
        def simple_objective(x, y):
            return {'mean': np.mean(x)}, np.mean(x)
        
        self.objective_func = simple_objective
    
    def test_is_improvement_maximize(self):
        """Test _is_improvement in maximize mode."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='maximize'
        )
        climber.state.best_objective = 5.0
        self.assertTrue(climber._is_improvement(6.0))
        self.assertFalse(climber._is_improvement(4.0))
    
    def test_is_improvement_minimize(self):
        """Test _is_improvement in minimize mode."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='minimize'
        )
        climber.state.best_objective = 5.0
        self.assertTrue(climber._is_improvement(4.0))
        self.assertFalse(climber._is_improvement(6.0))
    
    def test_is_improvement_target(self):
        """Test _is_improvement in target mode."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='target',
            target_value=10.0
        )
        climber.state.best_objective = 7.0  # distance = 3.0
        climber.state.best_distance = 3.0
        self.assertTrue(climber._is_improvement(8.0))  # distance = 2.0
        self.assertFalse(climber._is_improvement(5.0))  # distance = 5.0
    
    def test_calculate_delta_maximize(self):
        """Test _calculate_delta in maximize mode."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='maximize'
        )
        climber.state.current_objective = 5.0
        self.assertEqual(climber._calculate_delta(6.0), 1.0)
        self.assertEqual(climber._calculate_delta(4.0), -1.0)
    
    def test_calculate_delta_minimize(self):
        """Test _calculate_delta in minimize mode."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='minimize'
        )
        climber.state.current_objective = 5.0
        self.assertEqual(climber._calculate_delta(4.0), 1.0)
        self.assertEqual(climber._calculate_delta(6.0), -1.0)
    
    def test_calculate_delta_target(self):
        """Test _calculate_delta in target mode."""
        climber = HillClimber(
            data=self.data,
            objective_func=self.objective_func,
            mode='target',
            target_value=10.0
        )
        climber.state.current_objective = 7.0  # distance = 3.0
        delta = climber._calculate_delta(8.0)  # new distance = 2.0
        self.assertEqual(delta, 1.0)
    
    def test_record_improvement(self):
        """Test record_improvement method via state."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func)
        climber.state.step = 1
        climber.state.best_objective = 5.0
        climber.state.best_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        climber.state.metrics = {'mean': 2.5}
        climber.state.history = {'Step': [], 'Objective value': [], 'Best_data': [], 'mean': []}
        
        climber.state.record_improvement()
        
        self.assertEqual(len(climber.state.history['Step']), 1)
        self.assertEqual(climber.state.history['Step'][0], 1)
        self.assertEqual(climber.state.history['Objective value'][0], 5.0)
        self.assertEqual(climber.state.history['mean'][0], 2.5)


class TestHillClimberClimb(unittest.TestCase):
    """Test cases for HillClimber climb method."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10)
        })
        
        def increasing_objective(x, y):
            # Objective that increases with mean
            mean_val = np.mean(x) + np.mean(y)
            return {'mean': mean_val}, mean_val
        
        self.objective_func = increasing_objective
    
    def test_climb_returns_tuple(self):
        """Test that climb returns a tuple of (best_data, steps_df)."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        result = climber.climb()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_climb_returns_dataframe_for_steps(self):
        """Test that climb returns a DataFrame for steps."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        best_data, steps_df = climber.climb()
        self.assertIsInstance(steps_df, pd.DataFrame)
    
    def test_climb_preserves_data_format(self):
        """Test that climb preserves input data format."""
        # Test with DataFrame
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        best_data, _ = climber.climb()
        self.assertIsInstance(best_data, pd.DataFrame)
        
        # Test with array
        climber_array = HillClimber(data=self.data.values, objective_func=self.objective_func, max_time=0.01)
        best_data_array, _ = climber_array.climb()
        self.assertIsInstance(best_data_array, np.ndarray)
    
    def test_climb_creates_steps_with_metrics(self):
        """Test that climb creates steps DataFrame with metric columns."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        _, steps_df = climber.climb()
        
        self.assertIn('Step', steps_df.columns)
        self.assertIn('Objective value', steps_df.columns)
        self.assertIn('Best_data', steps_df.columns)
        self.assertIn('mean', steps_df.columns)
    
    def test_climb_with_different_modes(self):
        """Test climb with different optimization modes."""
        climber_max = HillClimber(data=self.data, objective_func=self.objective_func, 
                                 max_time=0.01, mode='maximize')
        best_max, _ = climber_max.climb()
        
        climber_min = HillClimber(data=self.data, objective_func=self.objective_func, 
                                 max_time=0.01, mode='minimize')
        best_min, _ = climber_min.climb()
        
        self.assertIsNotNone(best_max)
        self.assertIsNotNone(best_min)


class TestHillClimberClimbParallel(unittest.TestCase):
    """Test cases for HillClimber climb_parallel method."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10)
        })
        # Use module-level function for parallel tests
        self.objective_func = _simple_objective_for_parallel
    
    def test_climb_parallel_returns_dict(self):
        """Test that climb_parallel returns a dictionary with correct structure."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        results = climber.climb_parallel(replicates=2)
        self.assertIsInstance(results, dict)
        self.assertIn('input_data', results)
        self.assertIn('results', results)
        self.assertEqual(len(results['results']), 2)
    
    def test_climb_parallel_result_structure(self):
        """Test that each result has correct structure."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        results = climber.climb_parallel(replicates=2)
        
        # Check dictionary structure
        self.assertIsInstance(results, dict)
        self.assertIn('input_data', results)
        self.assertIn('results', results)
        
        # Check input_data
        self.assertIsNotNone(results['input_data'])
        
        # Check results list
        for noisy_initial, best_data, steps_df in results['results']:
            self.assertIsInstance(steps_df, pd.DataFrame)
            self.assertIn('Step', steps_df.columns)
            self.assertIn('Objective value', steps_df.columns)
            # Check noisy_initial is returned
            self.assertIsNotNone(noisy_initial)
    
    def test_climb_parallel_saves_file(self):
        """Test that climb_parallel saves results to file."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            climber.climb_parallel(replicates=2, output_file=tmp_path)
            
            self.assertTrue(os.path.exists(tmp_path))
            
            with open(tmp_path, 'rb') as f:
                package = pickle.load(f)
            
            self.assertIn('results', package)
            self.assertIn('hyperparameters', package)
            self.assertIn('input_data', package)
            self.assertEqual(len(package['results']), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_climb_parallel_validates_cpu_count(self):
        """Test that climb_parallel validates replicate count against CPUs."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        
        with self.assertRaises(ValueError):
            climber.climb_parallel(replicates=cpu_count() + 100)
    
    def test_climb_parallel_with_initial_noise(self):
        """Test climb_parallel with initial noise parameter."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func, max_time=0.01)
        results = climber.climb_parallel(replicates=2, initial_noise=0.1)
        self.assertEqual(len(results), 2)


class TestHillClimberPlottingMethods(unittest.TestCase):
    """Test cases for HillClimber plotting methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10)
        })
        
        def simple_objective(x, y):
            return {'mean': np.mean(x)}, np.mean(x)
        
        self.objective_func = simple_objective
    
    def test_plot_input_method_exists(self):
        """Test that plot_input method exists."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func)
        self.assertTrue(hasattr(climber, 'plot_input'))
        self.assertTrue(callable(climber.plot_input))
    
    def test_plot_results_method_exists(self):
        """Test that plot_results method exists."""
        climber = HillClimber(data=self.data, objective_func=self.objective_func)
        self.assertTrue(hasattr(climber, 'plot_results'))
        self.assertTrue(callable(climber.plot_results))


if __name__ == '__main__':
    unittest.main()
