# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, MultiAgentWrapper, OptimOption, SimpleMultiHeadModel


# model input and output dimensions
PRODUCER_IN_DIM = 32
PRODUCER_OUT_DIM = 10
CONSUMER_IN_DIM = 32
CONSUMER_OUT_DIM = 100


def get_dqn_agent(in_dim, out_dim, config):
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(input_dim=in_dim, output_dim=out_dim, **config["model"]),
        optim_option=OptimOption(**config["optimization"])
    )
    return DQN(q_model, DQNConfig(**config["algorithm"]), **config["experience_memory"])


def get_sc_agents(agent_info_list, config):
    producer_agents = {
        f"producer.{info.id}": get_dqn_agent(PRODUCER_IN_DIM, PRODUCER_OUT_DIM, config)
        for info in agent_info_list
    }
    consumer_agents = {
        f"consumer.{info.id}": get_dqn_agent(CONSUMER_IN_DIM, CONSUMER_OUT_DIM, config)
        for info in agent_info_list
    }
    return MultiAgentWrapper({**producer_agents, **consumer_agents})
