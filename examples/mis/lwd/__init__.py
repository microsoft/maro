# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.utils.common import get_env
from maro.simulator import Env

from examples.mis.lwd.config import Config
from examples.mis.lwd.env_sampler.mis_env_sampler import MISEnvSampler, MISPlottingCallback
from examples.mis.lwd.simulator.mis_business_engine import MISBusinessEngine
from examples.mis.lwd.ppo import get_ppo_policy, get_ppo_trainer


config = Config()

# Environments
learn_env = Env(
    business_engine_cls=MISBusinessEngine,
    durations=config.max_tick,
    options={
        "graph_batch_size": config.train_graph_batch_size,
        "num_samples": config.train_num_samples,
        "device": torch.device(config.device),
        "num_node_lower_bound": config.num_node_lower_bound,
        "num_node_upper_bound": config.num_node_upper_bound,
        "node_sample_probability": config.node_sample_probability,
    },
)

test_env = Env(
    business_engine_cls=MISBusinessEngine,
    durations=config.max_tick,
    options={
        "graph_batch_size": config.eval_graph_batch_size,
        "num_samples": config.eval_num_samples,
        "device": torch.device(config.device),
        "num_node_lower_bound": config.num_node_lower_bound,
        "num_node_upper_bound": config.num_node_upper_bound,
        "node_sample_probability": config.node_sample_probability,
    },
)

# Agent, policy, and trainers
agent2policy = {agent: f"ppo_{agent}.policy" for agent in learn_env.agent_idx_list}

policies = [
    get_ppo_policy(
        name=f"ppo_{agent}.policy",
        state_dim=config.input_dim,
        action_num=config.output_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        init_lr=config.init_lr,
    )
    for agent in learn_env.agent_idx_list
]

trainers = [
    get_ppo_trainer(
        name=f"ppo_{agent}",
        state_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        init_lr=config.init_lr,
        clip_ratio=config.clip_ratio,
        max_tick=config.max_tick,
        batch_size=config.batch_size,
        reward_discount=config.reward_discount,
        graph_batch_size=config.train_graph_batch_size,
        graph_num_samples=config.train_num_samples,
        num_train_epochs=config.num_train_epochs,
        norm_base=config.reward_normalization_base,
    )
    for agent in learn_env.agent_idx_list
]

device_mapping = {f"ppo_{agent}.policy": config.device for agent in learn_env.agent_idx_list}

# Build RLComponentBundle
rl_component_bundle = RLComponentBundle(
    env_sampler=MISEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
        diversity_reward_coef=config.diversity_reward_coef,
        reward_normalization_base=config.reward_normalization_base,
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
    device_mapping=device_mapping,
    customized_callbacks=[MISPlottingCallback(log_dir=get_env("LOG_PATH", required=False, default="./"))],
)


__all__ = ["rl_component_bundle"]
