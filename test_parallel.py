#!/usr/bin/env python
"""Test script for parallel replica exchange."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from hill_climber import HillClimber

# Simple test objective function
def test_objective(x, y):
    """Maximize Pearson correlation."""
    corr, _ = pearsonr(x, y)
    if np.isnan(corr):
        corr = 0.0
    
    metrics = {'Pearson': corr}
    return metrics, corr

# Create test data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'x': np.random.rand(n),
    'y': np.random.rand(n)
})

print("Testing parallel replica exchange...")
print("=" * 60)

# Test sequential mode
print("\n1. Sequential mode (n_workers=0):")
climber_seq = HillClimber(
    data=data,
    objective_func=test_objective,
    max_time=0.5,  # 30 seconds
    n_replicas=4,
    exchange_interval=50,
    n_workers=0  # Sequential
)
best_seq, history_seq = climber_seq.climb()
print(f"Best objective: {history_seq['Objective value'].max():.6f}")
print(f"Total steps: {len(history_seq)}")

# Test parallel mode
print("\n2. Parallel mode (n_workers=4):")
climber_par = HillClimber(
    data=data,
    objective_func=test_objective,
    max_time=0.5,  # 30 seconds
    n_replicas=4,
    exchange_interval=50,
    n_workers=4  # Parallel
)
best_par, history_par = climber_par.climb()
print(f"Best objective: {history_par['Objective value'].max():.6f}")
print(f"Total steps: {len(history_par)}")

print("\n" + "=" * 60)
print("Tests completed successfully!")
print(f"Sequential best: {history_seq['Objective value'].max():.6f}")
print(f"Parallel best:   {history_par['Objective value'].max():.6f}")
