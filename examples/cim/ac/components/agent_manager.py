# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import Adam, RMSprop

from .agent import CIMAgent
from maro.rl import SimpleAgentManager, LearningModel, FullyConnectedNet, ActorCritic, ActorCriticHyperParameters
from maro.utils import set_seeds


def create_ac_agents(agent_id_list, config):
    num_actions = config.algorithm.num_actions
    set_seeds(config.seed)
    agent_dict = {}

    for agent_id in agent_id_list:
        policy_model = LearningModel(
            decision_layers=FullyConnectedNet(
                name=f'{agent_id}.policy', input_dim=config.algorithm.input_dim, output_dim=num_actions,
                activation=nn.Tanh, **config.algorithm.policy_model
            )
        )

        value_model = LearningModel(
            decision_layers=FullyConnectedNet(
                name=f'{agent_id}.value', input_dim=config.algorithm.input_dim, output_dim=1,
                activation=nn.LeakyReLU, **config.algorithm.value_model
            )
        )

        algorithm = ActorCritic(
            policy_model=policy_model,
            value_model=value_model,
            value_loss_func=nn.functional.smooth_l1_loss,
            policy_optimizer_cls=Adam,
            policy_optimizer_params=config.algorithm.policy_optimizer,
            value_optimizer_cls=RMSprop,
            value_optimizer_params=config.algorithm.value_optimizer,
            hyper_params=ActorCriticHyperParameters(
                num_actions=num_actions,
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
