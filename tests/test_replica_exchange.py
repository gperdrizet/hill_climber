"""Tests for replica exchange functionality."""
import numpy as np
import unittest
from hill_climber.replica_exchange import (
    TemperatureLadder,
    ExchangeStatistics,
    ExchangeScheduler,
    compute_exchange_probability,
    should_exchange
)


class TestTemperatureLadder(unittest.TestCase):
    """Tests for temperature ladder creation."""
    
    def test_geometric_ladder(self):
        """Test geometric temperature ladder."""
        ladder = TemperatureLadder.geometric(n_replicas=4, T_min=1.0, T_max=1000.0)
        
        self.assertEqual(len(ladder.temperatures), 4)
        self.assertEqual(ladder.temperatures[0], 1.0)
        self.assertAlmostEqual(ladder.temperatures[-1], 1000.0, places=5)
        self.assertTrue(np.all(np.diff(ladder.temperatures) > 0))
        self.assertEqual(ladder.n_replicas, 4)
    
    def test_linear_ladder(self):
        """Test linear temperature ladder."""
        ladder = TemperatureLadder.linear(n_replicas=5, T_min=10.0, T_max=50.0)
        
        self.assertEqual(len(ladder.temperatures), 5)
        self.assertEqual(ladder.temperatures[0], 10.0)
        self.assertEqual(ladder.temperatures[-1], 50.0)
        self.assertTrue(np.allclose(np.diff(ladder.temperatures), 10.0))


class TestExchangeStatistics(unittest.TestCase):
    """Tests for exchange statistics tracking."""
    
    def test_record_attempts(self):
        """Test recording exchange attempts."""
        stats = ExchangeStatistics(n_replicas=3)
        
        stats.record_attempt(0, 1, accepted=True)
        stats.record_attempt(0, 1, accepted=False)
        stats.record_attempt(1, 2, accepted=True)
        
        self.assertEqual(stats.attempts[0, 1], 2)
        self.assertEqual(stats.acceptances[0, 1], 1)
    
    def test_acceptance_rate(self):
        """Test acceptance rate calculation."""
        stats = ExchangeStatistics(n_replicas=4)
        
        stats.record_attempt(0, 1, accepted=True)
        stats.record_attempt(1, 2, accepted=True)
        stats.record_attempt(2, 3, accepted=False)
        stats.record_attempt(0, 1, accepted=False)
        
        overall = stats.get_overall_acceptance_rate()
        self.assertEqual(overall, 0.5)


class TestExchangeScheduler(unittest.TestCase):
    """Tests for exchange scheduling."""
    
    def test_even_odd_strategy(self):
        """Test even/odd exchange scheduling."""
        scheduler = ExchangeScheduler(n_replicas=4, strategy='even_odd')
        
        pairs1 = scheduler.get_pairs()
        self.assertEqual(pairs1, [(0, 1), (2, 3)])
        
        pairs2 = scheduler.get_pairs()
        self.assertEqual(pairs2, [(1, 2)])


class TestExchangeProbability(unittest.TestCase):
    """Tests for exchange probability calculations."""
    
    def test_favorable_exchange(self):
        """Test that favorable exchanges have prob >= 0.9."""
        prob = compute_exchange_probability(
            obj1=10, obj2=20, temp1=100, temp2=200, mode='maximize'
        )
        self.assertGreater(prob, 0.9)
    
    def test_unfavorable_exchange(self):
        """Test that unfavorable exchanges have 0<prob<1."""
        # For maximize: moving better objective to hotter temp creates moderate acceptance
        obj1 = 10  # Replica 1 has lower objective  
        obj2 = 15  # Replica 2 has higher objective (better for maximize)
        temp1 = 1.0  # Replica 1 at cooler temperature
        temp2 = 2.0  # Replica 2 at hotter temperature
        
        prob = compute_exchange_probability(obj1, obj2, temp1, temp2, 'maximize')
        self.assertGreater(prob, 0)
        self.assertLess(prob, 1)


if __name__ == '__main__':
    unittest.main()
