# Virtual Machine Scheduling

A virtual machine (VM) scheduler is a cloud computing service component responsible for providing compute resources to satisfy user demands. A good resource allocation policy should aim to optimize several metrics at the same time, such as user wait time, profit, energy consumption and physical machine (PM) overload. Many commercial cloud providers use rule-based policies. Alternatively, the policy can also be optimized using reinforcement learning (RL) techniques, which involves simulating with historical data. This example demonstrates how DQN and Actor-Critic algorithms can be applied to this scenario. In this folder, you can find:  

* ``config.py``, which contains general configurations for the scenario;
* ``algorithms``, which contains configurations for the Actor-Critic, DQN algorithms, including network configurations;
* ``env_sampler.py``, which defines state, action and reward shaping in the ``VMEnvSampler`` class;
* ``policy_trainer.py``, which contains a registry that for the policies and algorithms defined in ``algorithms``;
* ``callbacks.py``, which defines routines to be invoked at the end of training or evaluation episodes.

See ``README.md`` under ``examples/rl`` for details about running the single-threaded learning workflow. We recommend that you follow this example to write your own scenarios.


# Some Comments About the Results

This example is meant to serve as a demonstration of using MARO's RL toolkit in a real-life scenario. In fact, we have yet to find a configuration that makes the policy learned by either DQN or Actor-Critic perform reasonably well in our experimental settings.

For reference, the best results have been achieved by the ``Best Fit`` algorithm (see ``examples/vm_scheduling/rule_based_algorithm/best_fit.py`` for details). The over-subscription rate is 115% in the over-subscription settings.

|Topology | PM Setting | Time Spent(s) | Total VM Requests |Successful Allocation| Energy Consumption| Total Oversubscriptions | Total Overload PMs
|:----:|-----|:--------:|:---:|:-------:|:----:|:---:|:---:|
|10k| 100 PMs, 32 Cores, 128 GB  | 104.98|10,000| 10,000| 2,399,610 | 0 | 0|
|10k.oversubscription| 100 PMs, 32 Cores, 128 GB|  101.00 |10,000 |10,000| 2,386,371| 279,331 | 0|
|336k| 880 PMs, 16 Cores, 112 GB | 7,896.37 |335,985| 109,249 |26,425,878 | 0 | 0 |
|336k.oversubscription| 880 PMs, 16 Cores, 112 GB | 7,903.33| 335,985| 115,008 | 27,440,946 | 3,868,475 | 0
