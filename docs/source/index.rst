What is MARO?
===============

.. figure:: ./images/logo.svg
    :width: 666px
    :align: center
    :alt: MARO
    :target: https://github.com/microsoft/maro

Multi-Agent Resource Optimization (MARO) platform is an instance of Reinforcement
learning as a Service (RaaS) for real-world resource optimization. It can be
applied to many important industrial domains, such as container inventory
management in logistics, bike repositioning in transportation, virtual machine
provisioning in data centers, and asset management in finance. Besides
`Reinforcement Learning <https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf>`_ (RL), it
also supports other planning/decision mechanisms, such as
`Operations Research <https://en.wikipedia.org/wiki/Operations_research>`_.

Key Components
---------------

.. figure:: ./images/maro_overview.svg
   :width: 1000px

- Simulation toolkit: it provides some predefined scenarios, and the reusable wheels for building new scenarios.
- RL toolkit: it provides a full-stack abstraction for RL, such as agent manager, agent, RL algorithms, learner, actor, and various shapers.
- Distributed toolkit: it provides distributed communication components, interface of user-defined functions for message auto-handling, cluster provision, and job orchestration.

Quick Start
-------------

.. code-block:: python

    from maro.simulator import Env
    from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent

    from random import randint

    # Initialize an Env for cim scenario
    env = Env(scenario="cim", topology="toy.5p_ssddd_l0.0", start_tick=0, durations=100)

    metrics: object = None
    decision_event: DecisionEvent = None
    is_done: bool = False
    action: Action = None

    # Start the env with a None Action
    metrics, decision_event, is_done = env.step(None)

    while not is_done:
        # Generate a random Action according to the action_scope in DecisionEvent
        action_scope = decision_event.action_scope
        to_discharge = action_scope.discharge > 0 and randint(0, 1) > 0

        action = Action(
            decision_event.vessel_idx,
            decision_event.port_idx,
            randint(0, action_scope.discharge if to_discharge else action_scope.load),
            ActionType.DISCHARGE if to_discharge else ActionType.LOAD
        )

        # Respond the environment with the generated Action
        metrics, decision_event, is_done = env.step(action)

Contents
----------

.. toctree::
    :maxdepth: 2
    :caption: Installation

    installation/pip_install.rst
    installation/playground.rst
    installation/grass_azure_cluster_provisioning.rst
    installation/grass_on_premises_cluster_provisioning.rst
    installation/k8s_aks_cluster_provisioning.rst
    installation/multi_processes_localhost_provisioning.rst

.. toctree::
    :maxdepth: 2
    :caption: Scenarios

    scenarios/container_inventory_management.rst
    scenarios/citi_bike.rst
    scenarios/vm_scheduling.rst
    scenarios/command_line.rst

.. toctree::
    :maxdepth: 2
    :caption: Examples

    examples/greedy_policy_citi_bike.rst

.. toctree::
    :maxdepth: 2
    :caption: Key Components

    key_components/simulation_toolkit.rst
    key_components/data_model.rst
    key_components/event_buffer.rst
    key_components/business_engine.rst
    key_components/rl_toolkit.rst
    key_components/distributed_toolkit.rst
    key_components/communication.rst
    key_components/orchestration.rst
    key_components/dashboard_visualization.rst
    key_components/geographic_visualization.rst

.. toctree::
    :maxdepth: 2
    :caption: API Documents

    apidoc/maro.rst
