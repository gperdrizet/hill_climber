"""Integration tests using actual objective functions from the notebook."""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hill_climber import HillClimber


def objective_spearman_large_pearson_small(x, y):
    """Maximize Spearman correlation while minimizing Pearson correlation."""
    pearson_corr = pearsonr(x, y)[0]
    spearman_corr = spearmanr(x, y)[0]
    objective = abs(spearman_corr) - abs(pearson_corr)
    
    metrics = {
        'Pearson coefficient': pearson_corr,
        'Spearman coefficient': spearman_corr
    }
    
    return metrics, objective


def objective_3d_simple(x, y, z):
    """Simple 3D objective: maximize sum of means."""
    total = np.mean(x) + np.mean(y) + np.mean(z)
    return {'total_mean': total}, total


class TestIntegrationWithRealObjective(unittest.TestCase):
    """Integration tests using real objective functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20)
        })
    
    def test_climb_with_real_objective(self):
        """Test climb() with actual correlation objective."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_spread=0.1,
            mode='maximize'
        )
        
        best_data, steps_df = climber.climb()
        
        self.assertIsInstance(best_data, pd.DataFrame)
        self.assertIsInstance(steps_df, pd.DataFrame)
        self.assertIn('Pearson coefficient', steps_df.columns)
        self.assertIn('Spearman coefficient', steps_df.columns)
        self.assertIn('Objective value', steps_df.columns)
    
    def test_climb_with_simulated_annealing(self):
        """Test climb() with simulated annealing enabled."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_spread=0.1,
            temperature=10.0,
            cooling_rate=0.005,
            mode='maximize'
        )
        
        best_data, steps_df = climber.climb()
        
        self.assertIsNotNone(best_data)
        self.assertGreater(len(steps_df), 0)
    
    # climb_parallel removed in v2.0 (replaced by replica exchange in climb())
    # def test_climb_parallel_with_real_objective(self):
    
    def test_minimize_mode_with_real_objective(self):
        """Test minimize mode with real objective function."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_spread=0.1,
            mode='minimize'
        )
        
        best_data, steps_df = climber.climb()
        
        self.assertIsNotNone(best_data)
        self.assertGreater(len(steps_df), 0)
    
    def test_target_mode_with_real_objective(self):
        """Test target mode with real objective function."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_spread=0.1,
            mode='target',
            target_value=0.5
        )
        
        best_data, steps_df = climber.climb()
        
        self.assertIsNotNone(best_data)
        self.assertGreater(len(steps_df), 0)
    
    def test_data_format_preservation(self):
        """Test that optimization preserves DataFrame format and columns."""
        original_columns = self.data.columns.tolist()
        
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_spread=0.1
        )
        
        best_data, _ = climber.climb()
        
        self.assertIsInstance(best_data, pd.DataFrame)
        self.assertListEqual(list(best_data.columns), original_columns)
        self.assertEqual(best_data.shape, self.data.shape)


class TestIntegrationWithNDimensionalData(unittest.TestCase):
    """Integration tests with n-dimensional data."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create 3D data
        self.data_3d = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20),
            'z': np.random.rand(20)
        })
        
        # Create 4D data
        self.data_4d = np.random.rand(15, 4)
    
    def test_climb_with_3d_data(self):
        """Test climb() with 3D data."""
        climber = HillClimber(
            data=self.data_3d,
            objective_func=objective_3d_simple,
            max_time=0.02,
            step_spread=0.1,
            mode='maximize'
        )
        
        best_data, steps_df = climber.climb()
        
        self.assertIsInstance(best_data, pd.DataFrame)
        self.assertEqual(best_data.shape[1], 3)
        self.assertIn('total_mean', steps_df.columns)
    
    def test_climb_with_4d_numpy_array(self):
        """Test climb() with 4D numpy array data."""
        def objective_4d(a, b, c, d):
            variance = np.var(a) + np.var(b) + np.var(c) + np.var(d)
            return {'total_variance': variance}, variance
        
        climber = HillClimber(
            data=self.data_4d,
            objective_func=objective_4d,
            max_time=0.02,
            step_spread=0.1,
            mode='maximize'
        )
        
        best_data, steps_df = climber.climb()
        
        self.assertIsInstance(best_data, np.ndarray)
        self.assertEqual(best_data.shape[1], 4)
        self.assertIn('total_variance', steps_df.columns)
    
    # climb_parallel removed in v2.0 (replaced by replica exchange in climb())
    # def test_climb_parallel_with_3d_data(self):


if __name__ == '__main__':
    unittest.main()
