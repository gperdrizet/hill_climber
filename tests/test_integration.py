"""Integration tests using actual objective functions from the notebook."""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add package directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hill_climber import HillClimber


# Real objective function from notebook
def objective_spearman_large_pearson_small(x, y):
    """Maximize Spearman correlation while minimizing Pearson correlation."""
    pearson_corr = pd.Series(x).corr(pd.Series(y), method='pearson')
    spearman_corr = pd.Series(x).corr(pd.Series(y), method='spearman')
    objective = abs(spearman_corr) - abs(pearson_corr)
    
    metrics = {
        'Pearson coefficient': pearson_corr,
        'Spearman coefficient': spearman_corr
    }
    
    return metrics, objective


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
            max_time=0.02,  # Very short for testing
            step_size=0.1,
            mode='maximize'
        )
        
        best_data, steps_df = climber.climb()
        
        # Verify structure
        self.assertIsInstance(best_data, pd.DataFrame)
        self.assertIsInstance(steps_df, pd.DataFrame)
        
        # Verify metrics columns exist
        self.assertIn('Pearson coefficient', steps_df.columns)
        self.assertIn('Spearman coefficient', steps_df.columns)
        self.assertIn('Objective value', steps_df.columns)
    
    def test_climb_with_simulated_annealing(self):
        """Test climb() with simulated annealing enabled."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_size=0.1,
            temperature=10.0,
            cooling_rate=0.995,
            mode='maximize'
        )
        
        best_data, steps_df = climber.climb()
        
        # Should complete without errors
        self.assertIsNotNone(best_data)
        self.assertGreater(len(steps_df), 0)
    
    def test_climb_parallel_with_real_objective(self):
        """Test climb_parallel() with real objective function."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_size=0.1,
            temperature=5.0,
            cooling_rate=0.999,
            mode='maximize'
        )
        
        results = climber.climb_parallel(
            replicates=2,
            initial_noise=0.05
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        for best_data, steps_df in results:
            self.assertIsInstance(best_data, pd.DataFrame)
            self.assertIsInstance(steps_df, pd.DataFrame)
            self.assertIn('Pearson coefficient', steps_df.columns)
            self.assertIn('Spearman coefficient', steps_df.columns)
    
    def test_minimize_mode_with_real_objective(self):
        """Test minimize mode with real objective function."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_size=0.1,
            mode='minimize'
        )
        
        best_data, steps_df = climber.climb()
        
        # Should complete successfully
        self.assertIsNotNone(best_data)
        self.assertGreater(len(steps_df), 0)
    
    def test_target_mode_with_real_objective(self):
        """Test target mode with real objective function."""
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_size=0.1,
            mode='target',
            target_value=0.5
        )
        
        best_data, steps_df = climber.climb()
        
        # Should complete successfully
        self.assertIsNotNone(best_data)
        self.assertGreater(len(steps_df), 0)
    
    def test_data_format_preservation(self):
        """Test that optimization preserves DataFrame format and columns."""
        original_columns = self.data.columns.tolist()
        
        climber = HillClimber(
            data=self.data,
            objective_func=objective_spearman_large_pearson_small,
            max_time=0.02,
            step_size=0.1
        )
        
        best_data, _ = climber.climb()
        
        # Verify format preservation
        self.assertIsInstance(best_data, pd.DataFrame)
        self.assertListEqual(list(best_data.columns), original_columns)
        self.assertEqual(best_data.shape, self.data.shape)


if __name__ == '__main__':
    unittest.main()
