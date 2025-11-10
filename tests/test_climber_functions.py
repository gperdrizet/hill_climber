"""Unit tests for helper functions in climber_functions module."""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add package directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hill_climber.climber_functions import perturb_vectors, extract_columns, calculate_correlation_objective


class TestPerturbVectors(unittest.TestCase):
    """Test cases for perturb_vectors function."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.array_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.df_data = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
    
    def test_perturb_array_returns_array(self):
        """Test that perturbing an array returns an array."""
        result = perturb_vectors(self.array_data, step_size=0.1)
        self.assertIsInstance(result, np.ndarray)
    
    def test_perturb_dataframe_returns_dataframe(self):
        """Test that perturbing a DataFrame returns a DataFrame."""
        result = perturb_vectors(self.df_data, step_size=0.1)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_perturb_preserves_shape(self):
        """Test that perturbation preserves data shape."""
        result = perturb_vectors(self.array_data, step_size=0.1)
        self.assertEqual(result.shape, self.array_data.shape)
    
    def test_perturb_preserves_columns(self):
        """Test that perturbation preserves DataFrame column names."""
        result = perturb_vectors(self.df_data, step_size=0.1)
        self.assertListEqual(list(result.columns), list(self.df_data.columns))
    
    def test_perturb_changes_data(self):
        """Test that perturbation actually changes the data."""
        np.random.seed(42)
        result = perturb_vectors(self.array_data, step_size=0.5)
        # At least one value should be different
        self.assertFalse(np.array_equal(result, self.array_data))
    
    def test_perturb_keeps_positive_values(self):
        """Test that all values remain positive after perturbation."""
        small_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        for _ in range(100):  # Test multiple times due to randomness
            result = perturb_vectors(small_data, step_size=0.05)
            self.assertTrue(np.all(result > 0), "All values should remain positive")
    
    def test_perturb_original_unchanged(self):
        """Test that original data is not modified."""
        original = self.array_data.copy()
        perturb_vectors(self.array_data, step_size=0.1)
        np.testing.assert_array_equal(self.array_data, original)


class TestExtractColumns(unittest.TestCase):
    """Test cases for extract_columns function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.df_data = pd.DataFrame({'a': [1.0, 3.0, 5.0], 'b': [2.0, 4.0, 6.0]})
    
    def test_extract_from_array(self):
        """Test extracting columns from numpy array."""
        x, y = extract_columns(self.array_data)
        np.testing.assert_array_equal(x, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(y, [2.0, 4.0, 6.0])
    
    def test_extract_from_dataframe(self):
        """Test extracting columns from DataFrame."""
        x, y = extract_columns(self.df_data)
        self.assertIsInstance(x, pd.Series)
        self.assertIsInstance(y, pd.Series)
        np.testing.assert_array_equal(x.values, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(y.values, [2.0, 4.0, 6.0])
    
    def test_extract_preserves_column_names(self):
        """Test that DataFrame column names are preserved."""
        x, y = extract_columns(self.df_data)
        self.assertEqual(x.name, 'a')
        self.assertEqual(y.name, 'b')
    
    def test_extract_returns_correct_length(self):
        """Test that extracted columns have correct length."""
        x, y = extract_columns(self.array_data)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)


class TestCalculateCorrelationObjective(unittest.TestCase):
    """Test cases for calculate_correlation_objective function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.df_data = pd.DataFrame({'x': [1.0, 3.0, 5.0], 'y': [2.0, 4.0, 6.0]})
        
        # Simple objective function for testing
        def simple_objective(x, y):
            return {'sum': sum(x) + sum(y)}, sum(x) + sum(y)
        
        self.simple_objective = simple_objective
    
    def test_calculate_with_array(self):
        """Test calculating objective with array data."""
        metrics, objective = calculate_correlation_objective(
            self.array_data, 
            self.simple_objective
        )
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(objective, (int, float))
        self.assertEqual(objective, 21.0)  # 1+3+5+2+4+6
    
    def test_calculate_with_dataframe(self):
        """Test calculating objective with DataFrame data."""
        metrics, objective = calculate_correlation_objective(
            self.df_data,
            self.simple_objective
        )
        self.assertIsInstance(metrics, dict)
        self.assertEqual(objective, 21.0)
    
    def test_calculate_returns_metrics(self):
        """Test that metrics dictionary is returned correctly."""
        metrics, _ = calculate_correlation_objective(
            self.array_data,
            self.simple_objective
        )
        self.assertIn('sum', metrics)
        self.assertEqual(metrics['sum'], 21.0)
    
    def test_calculate_with_correlation_objective(self):
        """Test with actual correlation-based objective function."""
        def pearson_objective(x, y):
            corr = pd.Series(x).corr(pd.Series(y), method='pearson')
            return {'pearson': corr}, corr
        
        metrics, objective = calculate_correlation_objective(
            self.array_data,
            pearson_objective
        )
        self.assertIn('pearson', metrics)
        self.assertAlmostEqual(objective, 1.0, places=5)  # Perfect correlation


if __name__ == '__main__':
    unittest.main()
