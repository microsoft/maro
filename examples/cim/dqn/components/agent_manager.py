# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import RMSprop

from maro.rl import (
    SimpleStore, DQN, DQNConfig, FullyConnectedBlock, NNStack, OptimizerOptions, SimpleAgentManager,
    SimpleMultiHeadedModel
)
from maro.utils import set_seeds


def create_dqn_agents(agent_id_list, config):
    num_actions = config.algorithm.num_actions
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        q_net = NNStack(
            "q_value",
            FullyConnectedBlock(
                input_dim=config.algorithm.input_dim,
                output_dim=num_actions,
                activation=nn.LeakyReLU,
                is_head=True,
                **config.algorithm.model
            )
        )
        learning_model = SimpleMultiHeadedModel(
            q_net, 
            optimizer_options=OptimizerOptions(cls=RMSprop, params=config.algorithm.optimizer)
        )
        agent_dict[agent_id] = DQN(
            agent_id, learning_model, DQNConfig(**config.algorithm.hyper_params, loss_cls=nn.SmoothL1Loss),
            experience_pool=SimpleStore(["state", "action", "reward", "next_state", "loss"])
        )

    return agent_dict


class DQNAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent):
        self._assert_train_mode()

        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.agent_dict[agent_id].store_experiences(exp)

        for agent in self.agent_dict.values():
            agent.train()
