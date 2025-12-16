"""Tests for HillClimber class with replica exchange."""
import numpy as np
import pandas as pd
import unittest
import tempfile
import os
from hill_climber import HillClimber


def simple_objective(x, y):
    """Simple test objective: maximize correlation."""
    corr = pd.Series(x).corr(pd.Series(y))
    return {'correlation': corr}, corr


def mean_objective(x, y):
    """Simple objective: maximize mean."""
    mean_val = np.mean(x) + np.mean(y)
    return {'mean': mean_val}, mean_val


class TestHillClimber(unittest.TestCase):
    """Test cases for HillClimber with replica exchange."""
    
    def test_initialization(self):
        """Test HillClimber initialization."""
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        climber = HillClimber(
            data=data,
            objective_func=simple_objective,
            max_time=1,
            n_replicas=2,
            verbose=False,
            db_enabled=False
        )
        
        self.assertEqual(climber.n_replicas, 2)
        self.assertEqual(climber.T_min, 0.0001)  # Default is 0.0001
        self.assertEqual(climber.T_max, 0.01)
        self.assertTrue(climber.is_dataframe)
        self.assertEqual(climber.column_names, ['x', 'y'])
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        data = np.array([[1, 4], [2, 5], [3, 6]])
        climber = HillClimber(
            data=data,
            objective_func=simple_objective,
            max_time=0.002,
            n_replicas=1,
            exchange_interval=10,
            verbose=False,
            db_enabled=False
        )
        
        best_data = climber.climb()
        
        self.assertEqual(best_data.shape, (3, 2))
        self.assertFalse(climber.is_dataframe)
    
    def test_replica_exchange_runs(self):
        """Test that replica exchange completes."""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20)
        })
        
        climber = HillClimber(
            data=data,
            objective_func=simple_objective,
            max_time=0.005,
            n_replicas=2,
            exchange_interval=10,
            verbose=False,
            db_enabled=False
        )
        
        best_data = climber.climb()
        
        self.assertEqual(best_data.shape, data.shape)


if __name__ == '__main__':
    unittest.main()
