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

OptimizerState (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hill_climber.OptimizerState
   :members: record_step, record_improvement, record_exchange, get_history_dataframe, get_acceptance_rate
   :show-inheritance:

Replica Exchange Components
----------------------------

TemperatureLadder
~~~~~~~~~~~~~~~~~

.. autoclass:: hill_climber.TemperatureLadder
   :members: n_replicas, geometric, linear, custom
   :show-inheritance:

ExchangeStatistics
~~~~~~~~~~~~~~~~~~

.. autoclass:: hill_climber.ExchangeStatistics
   :members: record_attempt, get_acceptance_rate, get_overall_acceptance_rate, get_pair_acceptance_rate
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
