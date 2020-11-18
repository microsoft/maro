# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import Adam, RMSprop

from .agent import CIMAgent
from maro.rl import (
    ActorCritic, ActorCriticConfig, FullyConnectedBlock, LearningModel, LearningModule,
    OptimizerOptions, SimpleAgentManager
)
from maro.utils import set_seeds


def create_ac_agents(agent_id_list, config):
    num_actions = config.algorithm.num_actions
    set_seeds(config.seed)
    agent_dict = {}

    for agent_id in agent_id_list:
        actor_module = LearningModule(
            "actor",
            [FullyConnectedBlock(
                input_dim=config.algorithm.input_dim,
                output_dim=num_actions,
                activation=nn.Tanh,
                is_head=True,
                **config.algorithm.actor_model
            )],
            optimizer_options=OptimizerOptions(cls=Adam, params=config.algorithm.optimizer)
        )

        critic_module = LearningModule(
            "critic",
            [FullyConnectedBlock(
                input_dim=config.algorithm.input_dim,
                output_dim=1,
                activation=nn.LeakyReLU,
                is_head=True,
                **config.algorithm.critic_model
            )],
            optimizer_options=OptimizerOptions(cls=Adam, params=config.algorithm.optimizer)
        )

        algorithm = ActorCritic(
            LearningModel(actor_module, critic_module),
            config=ActorCriticConfig(
                critic_loss_func=nn.functional.smooth_l1_loss,
                **config.algorithm.hyper_parameters,
            )
        )

        agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm)

    return agent_dict


class ACAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent: dict):
        for agent_id, exp in experiences_by_agent.items():
            if isinstance(exp, list):
                for trajectory in exp:
                    self.agent_dict[agent_id].train(trajectory["states"], trajectory["actions"], trajectory["rewards"])
            else:
                self.agent_dict[agent_id].train(exp["states"], exp["actions"], exp["rewards"])
