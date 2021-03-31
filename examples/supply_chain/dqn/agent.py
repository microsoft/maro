# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, MultiAgentWrapper, OptimOption, SimpleMultiHeadModel


# model input and output dimensions
MANUFACTURER_IN_DIM = 6
MANUFACTURER_OUT_DIM = 10
CONSUMER_IN_DIM = 8
CONSUMER_OUT_DIM = 100


def get_dqn_agent(in_dim, out_dim, config):
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(input_dim=in_dim, output_dim=out_dim, **config["model"]),
        optim_option=OptimOption(**config["optimization"])
    )
    return DQN(q_model, DQNConfig(**config["hyper_params"]))


def get_sc_agents(agent_ids, config):
    manufacturer_agents = {
        id_: get_dqn_agent(MANUFACTURER_IN_DIM, MANUFACTURER_OUT_DIM, config)
        for type_, id_ in agent_ids if type_ == "manufacture"
    }
    consumer_agents = {
        id_: get_dqn_agent(CONSUMER_IN_DIM, CONSUMER_OUT_DIM, config)
        for type_, id_ in agent_ids if type_ == "consumer"
    }
    return MultiAgentWrapper({**manufacturer_agents, **consumer_agents})
