# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

import numpy as np

import torch

from maro.rl import (
    DQN, DQNConfig, EpisodeBasedSchedule, FullyConnectedBlock, NullPolicy, OptimOption, QNetForDiscreteActionSpace,
    StepBasedSchedule, UniformSampler
)

from or_policy.minmax_policy import ConsumerMinMaxPolicy
from or_policy.eoq_policy import ConsumerEOQPolicy
from or_policy.base_policy import ProducerBaselinePolicy

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config


agent_ids = config["agent_ids"]
policy_ids = ["consumerstore", "consumer", "producer", "facility", "product", "productstore"]

config = config["policy"]

class SimpleQNet(QNetForDiscreteActionSpace):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component.forward(states)


def get_dqn_policy(cfg):
    q_net = SimpleQNet(
        FullyConnectedBlock(**cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"]),
        device=cfg["model"]["device"]
    )
    experience_manager = UniformSampler(**cfg["experience_manager"])
    return DQN(q_net, experience_manager, DQNConfig(**cfg["algorithm_config"]))

def get_base_consumer_policy(config):
    return ConsumerMinMaxPolicy(config)

def get_eoq_consumer_policy(config):
    return ConsumerEOQPolicy(config)

def get_base_producer_policy(config):
    return ProducerBaselinePolicy(config)

agent_to_policy = {agent_id: agent_id.split(".")[0] for agent_id in agent_ids}

null_policy = NullPolicy()

policy_dict = {
    'consumer': get_base_consumer_policy(config['consumer']),
    'producer': get_base_producer_policy(config['producer']),
    'consumerstore': get_dqn_policy(config['consumerstore']),
    'facility': null_policy,
    'product': null_policy,
    'productstore': null_policy
}

agent2policy = {agent_id: agent_id.split(".")[0] for agent_id in agent_ids}

# update schedules
schedule_type = {"step": StepBasedSchedule, "episode": EpisodeBasedSchedule}

def get_policy_update_schedule(cfg):
    return schedule_type[cfg["type"]](**cfg["args"])

# policy update schedule can be a dict or single EpisodeBasedSchedule or StepBasedSchedule.
# The latter indicates that all policies shared the same update schedule 
policy_update_schedule = {
    policy_id: get_policy_update_schedule(config[policy_id]["update_schedule"])
    for policy_id in policy_ids if policy_id in ['consumerstore']
}
