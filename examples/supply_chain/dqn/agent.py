# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, SimpleMultiHeadModel


def get_dqn_agent(config):
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(**config["model"]), optim_option=OptimOption(**config["optimization"])
    )
    return DQN(q_model, DQNConfig(**config["algorithm"]), **config["experience_memory"])
