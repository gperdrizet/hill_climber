"""Configuration dataclass for HillClimber optimizer.

This module provides a type-safe configuration object that encapsulates
all optimizer parameters with validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class OptimizerConfig:
    """Configuration for HillClimber optimizer.
    
    Attributes:
        objective_func: Function taking M column arrays, returns (metrics_dict, objective_value)
        mode: Optimization mode - 'maximize', 'minimize', or 'target'
        target_value: Target value (only used if mode='target')
        max_time: Maximum runtime in minutes
        step_spread: Perturbation spread as fraction of input range (default: 0.01 = 1%)
        perturb_fraction: Fraction of data points to perturb each step
        n_replicas: Number of replicas for parallel tempering (default: 4)
        T_min: Base temperature (will be used as T_min for ladder)
        T_max: Maximum temperature for hottest replica (default: 100 * T_min)
        cooling_rate: Temperature decay rate per successful step
        temperature_scheme: 'geometric' or 'linear' temperature spacing
        exchange_interval: Steps between exchange attempts
        exchange_strategy: 'even_odd', 'random', or 'all_neighbors'
        checkpoint_file: Path to save checkpoints (default: None, no checkpointing)
        checkpoint_interval: Batches between checkpoint saves (default: 1)
        db_enabled: Enable database logging for dashboard (default: False)
        db_path: Path to SQLite database file (default: 'data/hill_climber_progress.db')
        db_step_interval: Collect metrics every Nth step (default: exchange_interval // 1000)
        db_buffer_size: Number of pooled steps before database write (default: 10)
        verbose: Print progress messages (default: False)
        n_workers: Number of worker processes (default: n_replicas)
    """
    
    objective_func: Callable
    mode: str = 'maximize'
    target_value: Optional[float] = None
    max_time: float = 30.0
    step_spread: float = 0.01
    perturb_fraction: float = 0.001
    n_replicas: int = 4
    T_min: float = 0.1
    T_max: Optional[float] = None
    cooling_rate: float = 1e-8
    temperature_scheme: str = 'geometric'
    exchange_interval: int = 10000
    exchange_strategy: str = 'even_odd'
    checkpoint_file: Optional[str] = None
    checkpoint_interval: int = 1
    db_enabled: bool = False
    db_path: Optional[str] = None
    db_step_interval: Optional[int] = None
    db_buffer_size: int = 10
    verbose: bool = False
    n_workers: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode
        if self.mode not in ['maximize', 'minimize', 'target']:
            raise ValueError(
                f"mode must be 'maximize', 'minimize', or 'target', got '{self.mode}'"
            )
        
        # Validate target_value when mode is 'target'
        if self.mode == 'target' and self.target_value is None:
            raise ValueError(
                "target_value must be specified when mode='target'"
            )
        
        # Validate temperature scheme
        if self.temperature_scheme not in ['geometric', 'linear']:
            raise ValueError(
                f"temperature_scheme must be 'geometric' or 'linear', got '{self.temperature_scheme}'"
            )
        
        # Validate exchange strategy
        if self.exchange_strategy not in ['even_odd', 'random', 'all_neighbors']:
            raise ValueError(
                f"exchange_strategy must be 'even_odd', 'random', or 'all_neighbors', got '{self.exchange_strategy}'"
            )
        
        # Validate numeric ranges
        if self.max_time <= 0:
            raise ValueError(f"max_time must be positive, got {self.max_time}")
        
        if self.n_replicas <= 0:
            raise ValueError(f"n_replicas must be positive, got {self.n_replicas}")
        
        if not 0 < self.perturb_fraction <= 1:
            raise ValueError(
                f"perturb_fraction must be in (0, 1], got {self.perturb_fraction}"
            )
        
        if self.step_spread <= 0:
            raise ValueError(f"step_spread must be positive, got {self.step_spread}")
        
        if not 0 < self.cooling_rate < 1:
            raise ValueError(
                f"cooling_rate must be in (0, 1), got {self.cooling_rate}"
            )
        
        if self.T_min <= 0:
            raise ValueError(f"T_min must be positive, got {self.T_min}")
        
        if self.T_max is not None and self.T_max <= self.T_min:
            raise ValueError(
                f"T_max must be greater than T_min, got T_max={self.T_max}, T_min={self.T_min}"
            )
        
        # Validate objective function is callable
        if not callable(self.objective_func):
            raise ValueError("objective_func must be callable")
        
        # Set default T_max if not provided
        if self.T_max is None:
            self.T_max = self.T_min * 100
        
        # Set default db_path if db enabled but path not provided
        if self.db_enabled and self.db_path is None:
            self.db_path = 'data/hill_climber_progress.db'
        
        # Set default db_step_interval if db enabled but interval not provided
        if self.db_enabled and self.db_step_interval is None:
            self.db_step_interval = max(1, self.exchange_interval // 1000)
        
        # Set default n_workers if not provided
        if self.n_workers is None:
            self.n_workers = self.n_replicas
