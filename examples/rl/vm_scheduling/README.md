# Container Inventory Management

Container inventory management (CIM) is a scenario where reinforcement learning (RL) can potentially prove useful. In this folder you can find:
* ``env_wrapper.py``, which contains a function to generate an environment wrapper to interact
with our "agent" (see below);
* ``agent_wrapper.py``, which contains a function to generate an agent wrapper to interact
with the environment wrapper;
* ``policy_index``, which maps policy names to functions that create them; the functions to create DQN and Actor-Critic policies are defined in ``dqn.py`` and ``ac.py``, respectively.

The code for the actual learning workflows (e.g., learner, roll-out worker and trainer) can be found under ``examples/rl/workflows``. The reason for putting it in a separate folder is that these workflows apply to any scenario, so long as the necessary component generators, such as the ones listed above, are provided. See ``README`` under ``examples/rl`` for details. We recommend that you follow this example to write your own scenarios.