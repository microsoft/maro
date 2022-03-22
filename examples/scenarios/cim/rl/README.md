# Container Inventory Management

This example demonstrates the use of MARO's RL toolkit to optimize container inventory management. The scenario consists of a set of ports, each acting as a learning agent, and vessels that transfer empty containers among them. Each port must decide 1) whether to load or discharge containers when a vessel arrives and 2) how many containers to be loaded or discharged. The objective is to minimize the overall container shortage over a certain period of time. In this folder you can find:
* ``config.py``, which contains general configurations for the scenario;
* ``algorithms``, which contains configurations for the Actor-Critic, DQN and discrete-MADDPG algorithms, including network configurations;
* ``env_sampler.py``, which defines state, action and reward shaping in the ``CIMEnvSampler`` class;
* ``policy_trainer.py``, which contains a registry for the policies and algorithms defined in ``algorithms``;
* ``callbacks.py``, which defines routines to be invoked at the end of training or evaluation episodes.

See ``README.md`` under ``examples/rl`` for details about running the single-threaded learning workflow. We recommend that you follow this example to write your own scenarios.