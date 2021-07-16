# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath
from typing import List

import numpy as np
import torch

from maro.rl import (
    DQN, DQNConfig, FullyConnectedBlock, NullPolicy, OptimOption, DiscreteQNet, ExperienceManager
)

from examples.supply_chain.or_policy.minmax_policy import ConsumerMinMaxPolicy
from examples.supply_chain.or_policy.eoq_policy import ConsumerEOQPolicy
from examples.supply_chain.or_policy.base_policy import ProducerBaselinePolicy


class SimpleQNet(DiscreteQNet):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_dqn_policy(name: str, cfg):
    q_net = SimpleQNet(
        FullyConnectedBlock(**cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"]),
        device=cfg["model"]["device"]
    )
    dqn_policy = DQN(
        name=name,
        q_net=q_net,
        experience_manager=ExperienceManager(**cfg["experience_manager"], sampler_cls=None),
        config=DQNConfig(**cfg["algorithm_config"])
    )
    return dqn_policy

def get_base_consumer_policy(name: str, config: dict):
    return ConsumerMinMaxPolicy(name, config)

def get_eoq_consumer_policy(name: str, config: dict):
    return ConsumerEOQPolicy(name, config)

def get_base_producer_policy(name: str, config: dict):
    return ProducerBaselinePolicy(name, config)

def get_policy_mapping(config) -> (list, dict):
    # policy_ids = ["consumerstore", "consumer", "producer", "facility", "product", "productstore"]
    policies = [
        get_base_consumer_policy("consumer", config["policy"]["consumer"]),
        get_base_producer_policy("producer", config["policy"]["producer"]),
        get_dqn_policy("consumerstore", config["policy"]["consumerstore"]),
        NullPolicy(name="facility"),
        NullPolicy(name="product"),
        NullPolicy(name="productstore")
    ]

    agent2policy = {agent_id: agent_id.split(".")[0] for agent_id in config["agent_id_list"]}
    return policies, agent2policy

def get_replay_agent_ids(agent_id_list) -> List[str]:
    replay_agent_ids = [agent_id for agent_id in agent_id_list if agent_id.startswith("consumerstore")]
    return replay_agent_ids
