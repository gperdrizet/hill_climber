"""Configuration dataclass for HillClimber optimizer.

This module provides a type-safe configuration object that encapsulates
all optimizer parameters with validation, along with default values and
constants used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable


# =============================================================================
# Default Optimizer Parameters
# =============================================================================

# Temperature parameters
DEFAULT_T_MIN = 0.1  # Default minimum temperature for coldest replica
DEFAULT_T_MAX_MULTIPLIER = 100  # T_max = T_min * this multiplier when not specified
DEFAULT_COOLING_RATE = 1e-8  # Default temperature decay rate per step

# Perturbation parameters
DEFAULT_STEP_SPREAD = 0.01  # Default perturbation spread (1% of data range)
DEFAULT_PERTURB_FRACTION = 0.001  # Default fraction of points to perturb (0.1%)

# Replica exchange parameters
DEFAULT_N_REPLICAS = 4  # Default number of replicas for parallel tempering
DEFAULT_EXCHANGE_INTERVAL = 10000  # Default steps between exchange attempts
DEFAULT_TEMPERATURE_SCHEME = 'geometric'  # Default temperature ladder spacing
DEFAULT_EXCHANGE_STRATEGY = 'even_odd'  # Default replica pairing strategy

# Runtime parameters
DEFAULT_MAX_TIME = 30.0  # Default maximum runtime in minutes
DEFAULT_MODE = 'maximize'  # Default optimization mode

# Checkpointing parameters
DEFAULT_CHECKPOINT_INTERVAL = 1  # Default batches between checkpoint saves

# Database parameters
DEFAULT_DB_PATH = 'data/hill_climber_progress.db'  # Default database file path
DEFAULT_DB_BUFFER_SIZE = 10  # Default number of steps to buffer before write
DB_STEP_INTERVAL_DIVISOR = 1000  # Divisor for calculating default db_step_interval (0.1% sampling)

# =============================================================================
# Validation Constants
# =============================================================================

# Valid optimization modes
VALID_MODES = ['maximize', 'minimize', 'target']

# Valid temperature schemes
VALID_TEMPERATURE_SCHEMES = ['geometric', 'linear']

# Valid exchange strategies
VALID_EXCHANGE_STRATEGIES = ['even_odd', 'random', 'all_neighbors']

# =============================================================================
# Column Name Patterns
# =============================================================================

# Default column name prefix when data has no column names
DEFAULT_COLUMN_PREFIX = 'col_'


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
    mode: str = DEFAULT_MODE
    target_value: Optional[float] = None
    max_time: float = DEFAULT_MAX_TIME
    step_spread: float = DEFAULT_STEP_SPREAD
    perturb_fraction: float = DEFAULT_PERTURB_FRACTION
    n_replicas: int = DEFAULT_N_REPLICAS
    T_min: float = DEFAULT_T_MIN
    T_max: Optional[float] = None
    cooling_rate: float = DEFAULT_COOLING_RATE
    temperature_scheme: str = DEFAULT_TEMPERATURE_SCHEME
    exchange_interval: int = DEFAULT_EXCHANGE_INTERVAL
    exchange_strategy: str = DEFAULT_EXCHANGE_STRATEGY
    checkpoint_file: Optional[str] = None
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    db_enabled: bool = False
    db_path: Optional[str] = None
    db_step_interval: Optional[int] = None
    db_buffer_size: int = DEFAULT_DB_BUFFER_SIZE
    verbose: bool = False
    n_workers: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}, got '{self.mode}'"
            )
        
        # Validate target_value when mode is 'target'
        if self.mode == 'target' and self.target_value is None:
            raise ValueError(
                "target_value must be specified when mode='target'"
            )
        
        # Validate temperature scheme
        if self.temperature_scheme not in VALID_TEMPERATURE_SCHEMES:
            raise ValueError(
                f"temperature_scheme must be one of {VALID_TEMPERATURE_SCHEMES}, got '{self.temperature_scheme}'"
            )
        
        # Validate exchange strategy
        if self.exchange_strategy not in VALID_EXCHANGE_STRATEGIES:
            raise ValueError(
                f"exchange_strategy must be one of {VALID_EXCHANGE_STRATEGIES}, got '{self.exchange_strategy}'"
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
            self.T_max = self.T_min * DEFAULT_T_MAX_MULTIPLIER
        
        # Set default db_path if db enabled but path not provided
        if self.db_enabled and self.db_path is None:
            self.db_path = DEFAULT_DB_PATH
        
        # Set default db_step_interval if db enabled but interval not provided
        if self.db_enabled and self.db_step_interval is None:
            self.db_step_interval = max(1, self.exchange_interval // DB_STEP_INTERVAL_DIVISOR)
        
        # Set default n_workers if not provided
        if self.n_workers is None:
            self.n_workers = self.n_replicas
