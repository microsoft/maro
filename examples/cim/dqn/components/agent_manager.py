# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import RMSprop

from maro.rl import (
<<<<<<< HEAD
    ColumnBasedStore, DQN, DQNConfig, FullyConnectedBlock, LearningModel, NNStack, OptimizerOptions,
    AgentManager
=======
    ColumnBasedStore, DQN, DQNConfig, FullyConnectedBlock, NNStack, OptimizerOptions, SimpleAgentManager,
    SimpleMultiHeadedModel
>>>>>>> v0.2_merge_algorithm_into_agent
)
from maro.utils import set_seeds


def create_dqn_agents(agent_id_list, config):
    num_actions = config.algorithm.num_actions
    set_seeds(config.seed)
    agents = {}
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
<<<<<<< HEAD
        algorithm = DQN(
            learning_model,
            DQNConfig(**config.algorithm.hyper_params, loss_cls=nn.SmoothL1Loss)
        )
        agents[agent_id] = DQNAgent(
            agent_id, algorithm, ColumnBasedStore(**config.experience_pool),
            **config.training_loop_parameters
=======
        agent_dict[agent_id] = DQN(
            agent_id, learning_model, DQNConfig(**config.algorithm.hyper_params, loss_cls=nn.SmoothL1Loss),
            experience_pool=ColumnBasedStore(**config.experience_pool)
>>>>>>> v0.2_merge_algorithm_into_agent
        )

    return agents


class DQNAgentManager(AgentManager):
    def train(self, experiences_by_agent, performance=None):
        self._assert_train_mode()

        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.agents[agent_id].store_experiences(exp)

        for agent in self.agents.values():
            agent.train()
