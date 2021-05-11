# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

import numpy as np

import torch

from maro.rl import (
    DQN, DQNConfig, ExperienceMemory, FullyConnectedBlock, NullPolicy, OptimOption, QNetForDiscreteActionSpace,
    TrainingLoopConfig, get_sampler_cls
)

from or_policy.minmax_policy import ConsumerMinMaxPolicy
from or_policy.base_policy import ProducerBaselinePolicy

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config

agent_ids = config["agent_ids"]
config = config["policy"]



def get_base_consumer_policy(config):
    return ConsumerMinMaxPolicy(config)

def get_base_producer_policy(config):
    return ProducerBaselinePolicy(config)

# all consumers share the same underlying policy
policy_dict = {"consumer": get_base_consumer_policy(config["consumer"]),
               "producer": get_base_producer_policy(None)}

agent_to_policy = {agent_id: agent_id.split(".")[0] for agent_id in agent_ids}
