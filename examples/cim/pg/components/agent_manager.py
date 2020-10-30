# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import Adam

from .agent import CIMAgent
from maro.rl import SimpleAgentManager, LearningModel, FullyConnectedNet, PolicyGradient, \
    PolicyGradientHyperParameters
from maro.utils import set_seeds


def create_pg_agents(agent_id_list, config):
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

        algorithm = PolicyGradient(
            policy_model=policy_model,
            optimizer_cls=Adam,
            optimizer_params=config.algorithm.optimizer,
            hyper_params=PolicyGradientHyperParameters(
                num_actions=num_actions,
                **config.algorithm.hyper_parameters,
            )
        )

        agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm)

    return agent_dict


class PGAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent: dict):
        for agent_id, experiences in experiences_by_agent.items():
            if isinstance(experiences, list):
                for trajectory in experiences_by_agent:
                    self.agent_dict[agent_id].train(trajectory["states"], trajectory["actions"], trajectory["rewards"])
            else:
                self.agent_dict[agent_id].train(experiences["states"], experiences["actions"], experiences["rewards"])
