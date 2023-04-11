# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List

import numpy as np
from torch import nn

from maro.rl_v31.rl_component_bundle.rl_component_bundle import RLComponentBundle
from maro.rl_v31.training.algorithms.ppo import PPOTrainer
from maro.simulator import Env

from examples.cim.rl_v31.algorithms.ppo import get_ppo_critic, get_ppo_policy
from examples.cim.rl_v31.config import action_num, env_conf, reward_shaping_conf, state_dim
from examples.cim.rl_v31.env_wrapper import CimEnvWrapper, env

num_agents = len(env.agent_idx_list)

agent2policy = {agent: f"ppo_{agent}.policy" for agent in env.agent_idx_list}
policies = [get_ppo_policy(state_dim, action_num, f"ppo_{i}.policy") for i in range(num_agents)]
trainers = [
    PPOTrainer(
        name=f"ppo_{i}",
        memory_size=1000,
        critic_func=lambda: get_ppo_critic(state_dim),
        critic_loss_cls=nn.SmoothL1Loss,
        lam=0.0,
        reward_discount=0.0,
        clip_ratio=0.1,
        grad_iters=10,
    )
    for i in range(num_agents)
]


def metrics_agg_func(x: List[dict]) -> dict:
    keys = x[0].keys()
    ret = {}
    for key in keys:
        ret[key] = np.mean([e[key] for e in x])
    return ret


rl_component_bundle = RLComponentBundle(
    env_wrapper_func=lambda: CimEnvWrapper(Env(**env_conf), reward_eval_delay=reward_shaping_conf["time_window"]),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
    metrics_agg_func=metrics_agg_func,
)
