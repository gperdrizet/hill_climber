"""Replica exchange coordination and utilities."""
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TemperatureLadder:
    """Manages temperature ladder for replica exchange.
    
    Provides methods to create geometric, linear, or custom temperature
    schedules for parallel tempering optimization.
    """
    
    temperatures: np.ndarray
    
    @property
    def n_replicas(self) -> int:
        """Number of replicas in the ladder."""
        return len(self.temperatures)
    
    @classmethod
    def geometric(cls, n_replicas: int, T_min: float, T_max: float):
        """Create geometric temperature ladder.
        
        Args:
            n_replicas: Number of replicas
            T_min: Minimum (coldest) temperature
            T_max: Maximum (hottest) temperature
            
        Returns:
            TemperatureLadder instance
        """
        if n_replicas == 1:
            temps = np.array([T_min])
        else:
            ratio = (T_max / T_min) ** (1 / (n_replicas - 1))
            temps = T_min * (ratio ** np.arange(n_replicas))
        return cls(temperatures=temps)
    
    @classmethod
    def linear(cls, n_replicas: int, T_min: float, T_max: float):
        """Create linear temperature ladder.
        
        Args:
            n_replicas: Number of replicas
            T_min: Minimum (coldest) temperature
            T_max: Maximum (hottest) temperature
            
        Returns:
            TemperatureLadder instance
        """
        temps = np.linspace(T_min, T_max, n_replicas)
        return cls(temperatures=temps)
    
    @classmethod
    def custom(cls, temperatures: List[float]):
        """Create custom temperature ladder.
        
        Args:
            temperatures: List of temperatures (will be sorted)
            
        Returns:
            TemperatureLadder instance
        """
        return cls(temperatures=np.array(sorted(temperatures)))


class ExchangeScheduler:
    """Determines which replica pairs attempt exchanges each round."""
    
    def __init__(self, n_replicas: int, strategy: str = 'even_odd'):
        """Initialize scheduler.
        
        Args:
            n_replicas: Number of replicas
            strategy: Exchange strategy ('even_odd', 'random', 'all_neighbors')
        """

        self.n_replicas = n_replicas
        self.strategy = strategy
        self.round = 0
    
    def get_pairs(self) -> List[Tuple[int, int]]:
        """Get list of replica pairs to attempt exchange.
        
        Returns:
            List of (i, j) tuples where i < j
        """

        if self.strategy == 'even_odd':

            # Alternate between even and odd pairs for better mixing
            if self.round % 2 == 0:
                pairs = [(i, i+1) for i in range(0, self.n_replicas-1, 2)]

            else:
                pairs = [(i, i+1) for i in range(1, self.n_replicas-1, 2)]
        
        elif self.strategy == 'random':

            # Random pair selection
            indices = np.random.permutation(self.n_replicas)

            pairs = [(indices[i], indices[i+1]) 
                    for i in range(0, len(indices)-1, 2)]

            # Ensure i < j
            pairs = [(min(i,j), max(i,j)) for i, j in pairs]
        
        elif self.strategy == 'all_neighbors':

            # All neighboring pairs
            pairs = [(i, i+1) for i in range(self.n_replicas-1)]
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.round += 1

        return pairs


def compute_exchange_probability(obj1: float, obj2: float, 
                                 temp1: float, temp2: float,
                                 mode: str = 'maximize') -> float:
    """Compute probability of exchanging configurations.
    
    Uses Metropolis criterion for replica exchange:
    P(exchange) = min(1, exp(ΔE * Δβ))
    where ΔE depends on mode and Δβ = 1/T1 - 1/T2
    
    Args:
        obj1: Objective value of replica 1
        obj2: Objective value of replica 2
        temp1: Temperature of replica 1
        temp2: Temperature of replica 2
        mode: 'maximize', 'minimize', or 'target'
        
    Returns:
        Probability of accepting exchange (0 to 1)
    """
    # Convert to energy (lower is better)
    if mode == 'maximize':
        E1, E2 = -obj1, -obj2
    elif mode == 'minimize':
        E1, E2 = obj1, obj2
    else:
        # For target mode, energy is distance from target
        # This is handled at the state level, so treat as minimize
        E1, E2 = obj1, obj2
    
    delta_E = E2 - E1
    delta_beta = (1.0 / temp1) - (1.0 / temp2)
    
    # Metropolis criterion
    delta = delta_E * delta_beta
    
    if delta >= 0:
        return 1.0
    else:
        return np.exp(delta)


def should_exchange(obj1: float, obj2: float,
                   temp1: float, temp2: float,
                   mode: str = 'maximize') -> bool:
    """Determine if exchange should occur.
    
    Args:
        obj1: Objective value of replica 1
        obj2: Objective value of replica 2
        temp1: Temperature of replica 1  
        temp2: Temperature of replica 2
        mode: Optimization mode
        
    Returns:
        True if exchange should occur
    """
    prob = compute_exchange_probability(obj1, obj2, temp1, temp2, mode)
    return np.random.random() < prob
