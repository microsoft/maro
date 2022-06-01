# Container Inventory Management

This example demonstrates the use of MARO's RL toolkit to optimize container inventory management. The scenario consists of a set of ports, each acting as a learning agent, and vessels that transfer empty containers among them. Each port must decide 1) whether to load or discharge containers when a vessel arrives and 2) how many containers to be loaded or discharged. The objective is to minimize the overall container shortage over a certain period of time. In this folder you can find:
* ``__init__.py``, the entrance of this example. You must expose a `rl_component_bundle_cls` interface in `__init__.py` (see the example file for details);
* ``config.py``, which contains general configurations for the scenario;
* ``algorithms/``, which contains configurations for the PPO, Actor-Critic, DQN and discrete-MADDPG algorithms, including network configurations;
* ``rl_componenet_bundle.py``, which defines all necessary components to run a RL job. You can go through the doc string of `RLComponentBundle` for detailed explanation, or just read `CIMBundle` to learn its basic usage.

We recommend that you follow this example to write your own scenarios.
