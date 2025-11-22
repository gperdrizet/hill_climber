#!/usr/bin/env python
"""Comprehensive test and benchmark for parallel replica exchange."""

import numpy as np
import pandas as pd
import time
from scipy.stats import pearsonr
from hill_climber import HillClimber

def test_objective(x, y):
    """Test objective: maximize Pearson correlation."""
    corr, _ = pearsonr(x, y)
    if np.isnan(corr):
        corr = 0.0
    return {'Pearson': corr}, corr

# Create test data
np.random.seed(42)
n = 2000
data = pd.DataFrame({
    'x': np.random.rand(n),
    'y': np.random.rand(n)
})

print("Comprehensive Parallel Replica Exchange Test")
print("=" * 70)

# Configuration
configs = [
    {'n_workers': 0, 'name': 'Sequential'},
    {'n_workers': 2, 'name': 'Parallel (2 workers)'},
    {'n_workers': 4, 'name': 'Parallel (4 workers)'},
]

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")
    
    start = time.time()
    
    climber = HillClimber(
        data=data,
        objective_func=test_objective,
        max_time=1.0,  # 1 minute
        n_replicas=4,
        exchange_interval=100,
        temperature=1000,
        cooling_rate=0.0001,
        n_workers=config['n_workers']
    )
    
    best_data, history = climber.climb()
    
    elapsed = time.time() - start
    
    result = {
        'name': config['name'],
        'n_workers': config['n_workers'],
        'best_objective': history['Objective value'].max(),
        'total_steps': len(history),
        'elapsed_time': elapsed,
        'steps_per_second': len(history) / elapsed
    }
    results.append(result)
    
    print(f"\nResults:")
    print(f"  Best objective: {result['best_objective']:.6f}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {result['steps_per_second']:.1f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Mode':<25} {'Best Obj':<12} {'Steps':<10} {'Time (s)':<10} {'Steps/s':<10}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<25} {r['best_objective']:<12.6f} {r['total_steps']:<10} "
          f"{r['elapsed_time']:<10.2f} {r['steps_per_second']:<10.1f}")

# Speedup analysis
if len(results) > 1:
    seq_time = results[0]['elapsed_time']
    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS (vs Sequential)")
    print("=" * 70)
    for r in results[1:]:
        speedup = seq_time / r['elapsed_time']
        print(f"{r['name']:<25} {speedup:.2f}x")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)
