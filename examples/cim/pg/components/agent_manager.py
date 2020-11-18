# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import Adam

from .agent import CIMAgent
from maro.rl import (
    FullyConnectedBlock, LearningModel, LearningModule, OptimizerOptions, PolicyGradient, PolicyGradientConfig,
    SimpleAgentManager
)
from maro.utils import set_seeds


def create_pg_agents(agent_id_list, config):
    num_actions = config.algorithm.num_actions
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        policy_module = LearningModule(
            "policy",
            [FullyConnectedBlock(
                input_dim=config.algorithm.input_dim,
                output_dim=num_actions,
                activation=nn.Tanh,
                **config.algorithm.policy_model
            )],
            optimizer_options=OptimizerOptions(cls=Adam, params=config.algorithm.optimizer)
        )

        algorithm = PolicyGradient(
            model=LearningModel(policy_module),
            hyper_params=PolicyGradientConfig(**config.algorithm.hyper_parameters)
        )

        agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm)

    return agent_dict


class PGAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent: dict):
        for agent_id, exp in experiences_by_agent.items():
            if isinstance(exp, list):
                for trajectory in exp:
                    self.agent_dict[agent_id].train(trajectory["states"], trajectory["actions"],
                                                    trajectory["rewards"])
            else:
                self.agent_dict[agent_id].train(exp["states"], exp["actions"], exp["rewards"])
