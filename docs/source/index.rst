.. figure:: ./images/logo.svg
    :width: 250px
    :align: center
    :alt: MARO

MARO (Multi-Agent Resource Optimization) serves as a domain-specific Reinforcement
Learning (RL) solution. In MARO, multi-agent RL is used to solve real-world resource
optimization problems. It can be applied to many important industrial domains,
such as empty container repositioning (ECR) in logistics, bike repositioning in
transportation, virtual machine (VM) provisioning in data centers, and asset
management in finance, etc. Besides, MARO is not limited to RL, we also support
other planning/decision-related components, such as
`Operations Research (OR) <https://en.wikipedia.org/wiki/Operations_research>`_
based planning, which is already in our further worklist. MARO provides comprehensive
support in data processing, simulator building, RL algorithm selection, and
distributed training.


Key Components
====================
.. image:: ./images/maro_overview.svg
   :width: 1000px

- Simulation toolkit: it provides some predefined scenarios, and the reusable wheels for building new scenarios.
- RL toolkit: it provides a full-stack abstraction for RL, such as agent manager, agent, RL algorithms, learner, actor, and various shapers.
- Distributed toolkit: it provides distributed communication components, User-Defined Functions (UDF) interface for message auto-handling, cluster provision, and job orchestration.

Quick Start
====================
.. code-block:: python

    from maro.simulator import Env
    from maro.simulator.scenarios.ecr.common import Action

    start_tick = 0
    durations = 100  # 100 days

    # Initialize an environment with a specific scenario, related topology.
    env = Env(scenario="ecr", topology="5p_ssddd_l0.0",
            start_tick=start_tick, durations=durations)

    # Query environment summary, which includes business instances, intra-instance attributes, etc.
    print(env.summary)

    for ep in range(2):
        # Gym-like step function
        metrics, decision_event, is_done = env.step(None)

        while not is_done:
            past_week_ticks = [x for x in range(
                decision_event.tick - 7, decision_event.tick)]
            decision_port_idx = decision_event.port_idx
            intr_port_infos = ["booking", "empty", "shortage"]

            # Query the decision port booking, empty container inventory, shortage information in the past week
            past_week_info = env.snapshot_list["ports"][past_week_ticks:
                                                        decision_port_idx:
                                                        intr_port_infos]

            dummy_action = Action(decision_event.vessel_idx,
                                decision_event.port_idx, 0)

            # Drive environment with dummy action (no repositioning)
            metrics, decision_event, is_done = env.step(dummy_action)

        # Query environment business metrics at the end of an episode, it is your optimized object (usually includes multi-target).
        print(f"ep: {ep}, environment metrics: {env.get_metrics()}")
        env.reset()

Contents
====================
.. toctree::
    :maxdepth: 2
    :caption: Installation

    installation/pip_install.md
    installation/playground.md
    installation/grass_cluster_provisioning_on_azure.md
    installation/k8s_cluster_provisioning_on_azure.md

.. toctree::
    :maxdepth: 2
    :caption: Scenarios

    scenarios/ecr.md
    scenarios/citi_bike.md

.. toctree::
    :maxdepth: 2
    :caption: Examples

    examples/hello_world.md
    examples/ecr_single_host.md
    examples/ecr_distributed.md

.. toctree::
    :maxdepth: 2
    :caption: Key Components

    key_components/simulation_toolkit.md
    key_components/data_model.md
    key_components/event_buffer.md
    key_components/business_engine.md
    key_components/rl_toolkit.md
    key_components/distributed_toolkit.md
    key_components/communication.md
    key_components/orchestration.md

.. toctree::
    :maxdepth: 2
    :caption: Experiments
    
    experiments/ecr.md
    experiments/citi_bike.md

.. toctree::
    :maxdepth: 2
    :caption: API Documents

    apidoc/maro.rst