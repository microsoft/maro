# Container Inventory Management

This example demonstrates the use of MARO's RL toolkit to optimize container inventory management. The scenario consists of a set of ports, each acting as a learning agent, and vessels that transfer empty containers among them. Each port must decide 1) whether to load or discharge containers when a vessel arrives and 2) how many containers to be loaded or discharged. The objective is to minimize the overall container shortage over a certain period of time. In this folder you can find:
* ``config.py``, which contains environment and policy configurations for the scenario;
* ``env_sampler.py``, which defines state, action and reward shaping in the ``CIMEnvSampler`` class;
* ``policies.py``, which defines the Q-net for DQN and the network components for Actor-Critic;
* ``callbacks.py``, which defines routines to be invoked at the end of training or evaluation episodes.

The scripts for running the learning workflows can be found under ``examples/rl/workflows``. See ``README`` under ``examples/rl`` for details about the general applicability of these scripts. We recommend that you follow this example to write your own scenarios.