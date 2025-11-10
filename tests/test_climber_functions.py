"""Unit tests for helper functions in climber_functions module."""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add package directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hill_climber.climber_functions import perturb_vectors, extract_columns, calculate_correlation_objective, calculate_objective


class TestPerturbVectors(unittest.TestCase):
    """Test cases for perturb_vectors function."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.array_data_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.array_data_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.array_data_4d = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    
    def test_perturb_array_returns_array(self):
        """Test that perturbing an array returns an array."""
        result = perturb_vectors(self.array_data_2d, step_size=0.1)
        self.assertIsInstance(result, np.ndarray)
    
    def test_perturb_preserves_shape(self):
        """Test that perturbation preserves data shape."""
        result = perturb_vectors(self.array_data_2d, step_size=0.1)
        self.assertEqual(result.shape, self.array_data_2d.shape)
    
    def test_perturb_works_with_3d_data(self):
        """Test that perturbation works with 3D data."""
        result = perturb_vectors(self.array_data_3d, step_size=0.1)
        self.assertEqual(result.shape, self.array_data_3d.shape)
        self.assertIsInstance(result, np.ndarray)
    
    def test_perturb_works_with_4d_data(self):
        """Test that perturbation works with 4D data."""
        result = perturb_vectors(self.array_data_4d, step_size=0.1)
        self.assertEqual(result.shape, self.array_data_4d.shape)
        self.assertIsInstance(result, np.ndarray)
    
    def test_perturb_changes_data(self):
        """Test that perturbation actually changes the data."""
        np.random.seed(42)
        result = perturb_vectors(self.array_data_2d, step_size=0.5)
        # At least one value should be different
        self.assertFalse(np.array_equal(result, self.array_data_2d))
    
    def test_perturb_keeps_positive_values(self):
        """Test that all values remain positive after perturbation."""
        small_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        for _ in range(100):  # Test multiple times due to randomness
            result = perturb_vectors(small_data, step_size=0.05)
            self.assertTrue(np.all(result > 0), "All values should remain positive")
    
    def test_perturb_original_unchanged(self):
        """Test that original data is not modified."""
        original = self.array_data_2d.copy()
        perturb_vectors(self.array_data_2d, step_size=0.1)
        np.testing.assert_array_equal(self.array_data_2d, original)


class TestExtractColumns(unittest.TestCase):
    """Test cases for extract_columns function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array_data_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.array_data_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.array_data_4d = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    
    def test_extract_from_2d_array(self):
        """Test extracting columns from 2D numpy array."""
        x, y = extract_columns(self.array_data_2d)
        np.testing.assert_array_equal(x, [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(y, [2.0, 4.0, 6.0])
    
    def test_extract_from_3d_array(self):
        """Test extracting columns from 3D numpy array."""
        x, y, z = extract_columns(self.array_data_3d)
        np.testing.assert_array_equal(x, [1.0, 4.0, 7.0])
        np.testing.assert_array_equal(y, [2.0, 5.0, 8.0])
        np.testing.assert_array_equal(z, [3.0, 6.0, 9.0])
    
    def test_extract_from_4d_array(self):
        """Test extracting columns from 4D numpy array."""
        a, b, c, d = extract_columns(self.array_data_4d)
        np.testing.assert_array_equal(a, [1.0, 5.0])
        np.testing.assert_array_equal(b, [2.0, 6.0])
        np.testing.assert_array_equal(c, [3.0, 7.0])
        np.testing.assert_array_equal(d, [4.0, 8.0])
    
    def test_extract_returns_numpy_arrays(self):
        """Test that extracted columns are numpy arrays."""
        columns = extract_columns(self.array_data_2d)
        for col in columns:
            self.assertIsInstance(col, np.ndarray)
    
    def test_extract_returns_correct_length(self):
        """Test that extracted columns have correct length."""
        x, y = extract_columns(self.array_data_2d)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)
    
    def test_extract_returns_correct_number_of_columns(self):
        """Test that extract returns correct number of columns."""
        result_2d = extract_columns(self.array_data_2d)
        result_3d = extract_columns(self.array_data_3d)
        result_4d = extract_columns(self.array_data_4d)
        
        self.assertEqual(len(result_2d), 2)
        self.assertEqual(len(result_3d), 3)
        self.assertEqual(len(result_4d), 4)


class TestCalculateObjective(unittest.TestCase):
    """Test cases for calculate_objective and calculate_correlation_objective functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array_data_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.array_data_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        
        # Simple objective function for 2D testing
        def simple_objective_2d(x, y):
            return {'sum': sum(x) + sum(y)}, sum(x) + sum(y)
        
        # Simple objective function for 3D testing
        def simple_objective_3d(x, y, z):
            return {'sum': sum(x) + sum(y) + sum(z)}, sum(x) + sum(y) + sum(z)
        
        self.simple_objective_2d = simple_objective_2d
        self.simple_objective_3d = simple_objective_3d
    
    def test_calculate_with_2d_array(self):
        """Test calculating objective with 2D array data."""
        metrics, objective = calculate_objective(
            self.array_data_2d, 
            self.simple_objective_2d
        )
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(objective, (int, float))
        self.assertEqual(objective, 21.0)  # 1+3+5+2+4+6
    
    def test_calculate_with_3d_array(self):
        """Test calculating objective with 3D array data."""
        metrics, objective = calculate_objective(
            self.array_data_3d, 
            self.simple_objective_3d
        )
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(objective, (int, float))
        self.assertEqual(objective, 45.0)  # 1+4+7+2+5+8+3+6+9
    
    def test_calculate_returns_metrics(self):
        """Test that metrics dictionary is returned correctly."""
        metrics, _ = calculate_objective(
            self.array_data_2d,
            self.simple_objective_2d
        )
        self.assertIn('sum', metrics)
        self.assertEqual(metrics['sum'], 21.0)
    
    def test_calculate_with_correlation_objective(self):
        """Test with actual correlation-based objective function."""
        from scipy.stats import pearsonr
        
        def pearson_objective(x, y):
            corr = pearsonr(x, y)[0]
            return {'pearson': corr}, corr
        
        metrics, objective = calculate_objective(
            self.array_data_2d,
            pearson_objective
        )
        self.assertIn('pearson', metrics)
        self.assertAlmostEqual(objective, 1.0, places=5)  # Perfect correlation
    
    def test_backwards_compatibility_alias(self):
        """Test that calculate_correlation_objective is an alias for calculate_objective."""
        # Both should give same results
        metrics1, obj1 = calculate_objective(self.array_data_2d, self.simple_objective_2d)
        metrics2, obj2 = calculate_correlation_objective(self.array_data_2d, self.simple_objective_2d)
        
        self.assertEqual(metrics1, metrics2)
        self.assertEqual(obj1, obj2)


if __name__ == '__main__':
    unittest.main()
