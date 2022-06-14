# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.simulator import Env

from .algorithms.ac import get_ac, get_ac_policy
from .algorithms.dqn import get_dqn, get_dqn_policy
from .algorithms.maddpg import get_maddpg, get_maddpg_policy
from .algorithms.ppo import get_ppo, get_ppo_policy
from examples.cim.rl.config import action_num, algorithm, env_conf, reward_shaping_conf, state_dim
from examples.cim.rl.env_sampler import CIMEnvSampler

# Environments
learn_env = Env(**env_conf)
test_env = learn_env

# Agent, policy, and trainers
num_agents = len(learn_env.agent_idx_list)
agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}
if algorithm == "ac":
    policies = [get_ac_policy(state_dim, action_num, f"{algorithm}_{i}.policy") for i in range(num_agents)]
    trainers = [get_ac(state_dim, f"{algorithm}_{i}") for i in range(num_agents)]
elif algorithm == "ppo":
    policies = [get_ppo_policy(state_dim, action_num, f"{algorithm}_{i}.policy") for i in range(num_agents)]
    trainers = [get_ppo(state_dim, f"{algorithm}_{i}") for i in range(num_agents)]
elif algorithm == "dqn":
    policies = [get_dqn_policy(state_dim, action_num, f"{algorithm}_{i}.policy") for i in range(num_agents)]
    trainers = [get_dqn(f"{algorithm}_{i}") for i in range(num_agents)]
elif algorithm == "discrete_maddpg":
    policies = [get_maddpg_policy(state_dim, action_num, f"{algorithm}_{i}.policy") for i in range(num_agents)]
    trainers = [get_maddpg(state_dim, [1], f"{algorithm}_{i}") for i in range(num_agents)]
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

# Build RLComponentBundle
rl_component_bundle = RLComponentBundle(
    env_sampler=CIMEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
        reward_eval_delay=reward_shaping_conf["time_window"],
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
)
