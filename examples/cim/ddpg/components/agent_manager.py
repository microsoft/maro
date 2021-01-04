# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import RMSprop

from maro.rl import (
    ColumnBasedStore, DDPG, DDPGConfig, FullyConnectedBlock, GaussianNoiseExplorer, LearningModel, NNStack, 
    OptimizerOptions, SimpleAgentManager
)
from maro.utils import set_seeds

from .agent import DDPGAgent


def create_ddpg_agents(agent_id_list, config):
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        policy_net = NNStack(
            "policy",
            FullyConnectedBlock(
                input_dim=config.algorithm.input_dim,
                output_dim=1,
                activation=nn.LeakyReLU,
                is_head=True,
                **config.policy_model
            )
        )
        q_net = NNStack(
            "q_value",
            FullyConnectedBlock(
                input_dim=config.algorithm.input_dim + 1,
                output_dim=1,
                activation=nn.LeakyReLU,
                is_head=True,
                **config.q_value_model
            )
        )
        
        learning_model = LearningModel(
            policy_net, q_net, 
            optimizer_options={
                "policy": OptimizerOptions(cls=Adam, params=config.policy_optimizer),
                "q_value": OptimizerOptions(cls=RMSprop, params=config.q_value_optimizer)
            }
        )
        algorithm = DDPG(
            learning_model, DDPGConfig(**config.algorithm.hyper_params, loss_cls=nn.SmoothL1Loss),
            explorer=GaussianNoiseExplorer(min_action=config.min_action, max_action=config.max_action)
        )

        agent_dict[agent_id] = DDPGAgent(
            agent_id, algorithm, ColumnBasedStore(), **config.training_loop_parameters
        )

    return agent_dict


class DDPGAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent):
        self._assert_train_mode()

        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            self.agent_dict[agent_id].store_experiences(exp)
            self.agent_dict[agent_id].train()
