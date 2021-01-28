# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch.nn as nn
from torch.optim import Adam, RMSprop

from maro.rl import (
    AbsAgent, ActorCritic, ActorCriticConfig, FullyConnectedBlock, LearningModel, NNStack,
    OptimizerOptions, PolicyGradient, PolicyOptimizationConfig, SimpleAgentManager
)
from maro.utils import set_seeds


class POAgent(AbsAgent):
    def train(self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray):
        self._algorithm.train(states, actions, log_action_prob, rewards)


def create_po_agents(agent_id_list, config):
    input_dim, num_actions = config.input_dim, config.num_actions
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        actor_net = NNStack(
            "actor",
            FullyConnectedBlock(
                input_dim=input_dim,
                output_dim=num_actions,
                activation=nn.Tanh,
                is_head=True,
                **config.actor_model
            )
        )

        if config.type == "actor_critic":
            critic_net = NNStack(
                "critic",
                FullyConnectedBlock(
                    input_dim=config.input_dim,
                    output_dim=1,
                    activation=nn.LeakyReLU,
                    is_head=True,
                    **config.critic_model
                )
            )

            hyper_params = config.actor_critic_hyper_parameters
            hyper_params.update({"reward_discount": config.reward_discount})
            learning_model = LearningModel(
                actor_net, critic_net, 
                optimizer_options={
                    "actor": OptimizerOptions(cls=Adam, params=config.actor_optimizer),
                    "critic": OptimizerOptions(cls=RMSprop, params=config.critic_optimizer)
                } 
            )
            algorithm = ActorCritic(
                learning_model, ActorCriticConfig(critic_loss_func=nn.SmoothL1Loss(), **hyper_params)
            )
        else:
            learning_model = LearningModel(
                actor_net, 
                optimizer_options=OptimizerOptions(cls=Adam, params=config.actor_optimizer)  
            )
            algorithm = PolicyGradient(learning_model, PolicyOptimizationConfig(config.reward_discount))

        agent_dict[agent_id] = POAgent(name=agent_id, algorithm=algorithm)

    return agent_dict


class POAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent: dict):
        for agent_id, exp in experiences_by_agent.items():
            if not isinstance(exp, list):
                exp = [exp]
            for trajectory in exp:
                self.agent_dict[agent_id].train(
                    trajectory["state"],
                    trajectory["action"],
                    trajectory["log_action_probability"],
                    trajectory["reward"]
                )
