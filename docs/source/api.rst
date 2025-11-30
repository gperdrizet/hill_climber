API Reference
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

State Management
----------------

ReplicaState
~~~~~~~~~~~~

.. autoclass:: hill_climber.ReplicaState
   :members: to_dict, from_dict
   :show-inheritance:

.. autofunction:: hill_climber.create_replica_state

Replica Exchange Components
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

Core Functions
--------------

.. automodule:: hill_climber.climber_functions
   :members:
   :undoc-members:

Plotting Functions
------------------

.. automodule:: hill_climber.plotting_functions
   :members:
   :undoc-members:
