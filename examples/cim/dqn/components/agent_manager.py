# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import RMSprop

from maro.rl import (
    ColumnBasedStore, DQN, DQNConfig, EpsilonGreedyExplorer, FullyConnectedBlock, LearningModuleManager, LearningModule,
    OptimizerOptions, SimpleAgentManager
)
from maro.utils import set_seeds

from .agent import CIMAgent


def create_dqn_agents(agent_id_list, config):
    num_actions = config.algorithm.num_actions
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        q_module = LearningModule(
            "q_value",
            [FullyConnectedBlock(
                input_dim=config.algorithm.input_dim,
                output_dim=num_actions,
                activation=nn.LeakyReLU,
                is_head=True,
                **config.algorithm.model
            )],
            optimizer_options=OptimizerOptions(cls=RMSprop, params=config.algorithm.optimizer)
        )

        algorithm = DQN(
            model=LearningModuleManager(q_module),
            config=DQNConfig(
                **config.algorithm.config,
                loss_cls=nn.SmoothL1Loss,
                num_actions=num_actions
            )
        )

        experience_pool = ColumnBasedStore(**config.experience_pool)
        agent_dict[agent_id] = CIMAgent(
            agent_id, algorithm, EpsilonGreedyExplorer(num_actions), experience_pool,
            **config.training_loop_parameters
        )

    return agent_dict


class DQNAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent, performance=None):
        self._assert_train_mode()

        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self._agents[agent_id].store_experiences(exp)

        for agent in self._agents.values():
            agent.train()
