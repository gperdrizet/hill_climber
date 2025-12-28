API reference
=============

HillClimber
-----------

.. autoclass:: hill_climber.HillClimber
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

OptimizerConfig
~~~~~~~~~~~~~~~

.. autoclass:: hill_climber.OptimizerConfig
   :members:
   :undoc-members:
   :show-inheritance:

State management
----------------

ReplicaState
~~~~~~~~~~~~

.. autoclass:: hill_climber.ReplicaState
   :members: to_dict, from_dict
   :show-inheritance:

   Key attributes
   
   - **perturbation_num** (int): Global perturbation counter (monotonically increasing)
   - **num_accepted** (int): Number of SA-accepted steps
   - **num_improvements** (int): Number of improvements found
   - **best_data** (np.ndarray): Best solution found
   - **best_objective** (float): Best objective value
   - **best_metrics** (Dict): User-defined metrics at best solution
   - **current_data** (np.ndarray): Current solution being explored
   - **current_objective** (float): Current objective value
   - **temperature** (float): Current temperature

.. autofunction:: hill_climber.create_replica_state

Replica exchange components
----------------------------

TemperatureLadder
~~~~~~~~~~~~~~~~~

.. autoclass:: hill_climber.TemperatureLadder
   :members: n_replicas, geometric, linear, custom
   :show-inheritance:

ExchangeScheduler
~~~~~~~~~~~~~~~~~

.. autoclass:: hill_climber.ExchangeScheduler
   :members:
   :undoc-members:
   :show-inheritance:

Core functions
--------------

.. automodule:: hill_climber.climber_functions
   :members:
   :undoc-members:
