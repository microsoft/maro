# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.simulator import Env

from .algorithms.ac import get_ac, get_ac_policy
from .algorithms.dqn import get_dqn, get_dqn_policy
from examples.vm_scheduling.rl.config import algorithm, env_conf, num_features, num_pms, state_dim, test_env_conf
from examples.vm_scheduling.rl.env_sampler import VMEnvSampler

# Environments
learn_env = Env(**env_conf)
test_env = Env(**test_env_conf)

# Agent, policy, and trainers
action_num = num_pms + 1
agent2policy = {"AGENT": f"{algorithm}.policy"}
if algorithm == "ac":
    policies = [get_ac_policy(state_dim, action_num, num_features, f"{algorithm}.policy")]
    trainers = [get_ac(state_dim, num_features, algorithm)]
elif algorithm == "dqn":
    policies = [get_dqn_policy(state_dim, action_num, num_features, f"{algorithm}.policy")]
    trainers = [get_dqn(algorithm)]
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

# Build RLComponentBundle
rl_component_bundle = RLComponentBundle(
    env_sampler=VMEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
)
