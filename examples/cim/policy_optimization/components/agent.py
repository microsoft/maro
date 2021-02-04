# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import Adam, RMSprop

from maro.rl import (
    ActorCritic, ActorCriticConfig, FullyConnectedBlock, OptimOption, PolicyGradient, SimpleMultiHeadModel
)
from maro.utils import set_seeds


def create_po_agents(agent_id_list, config):
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        actor_net = FullyConnectedBlock(
            activation=nn.Tanh,
            is_head=True,
            **config.actor_model
        )

        if config.type == "actor_critic":
            critic_net = FullyConnectedBlock(
                activation=nn.LeakyReLU,
                is_head=True,
                **config.critic_model
            )

            hyper_params = config.actor_critic_hyper_parameters
            hyper_params.update({"reward_discount": config.reward_discount})
            learning_model = SimpleMultiHeadModel(
                {"actor": actor_net, "critic": critic_net}, 
                optim_option={
                    "actor": OptimOption(optim_cls=Adam, optim_params=config.actor_optimizer),
                    "critic": OptimOption(optim_cls=RMSprop, optim_params=config.critic_optimizer)
                }
            )
            agent_dict[agent_id] = ActorCritic(
                agent_id, learning_model, ActorCriticConfig(critic_loss_func=nn.SmoothL1Loss(), **hyper_params)
            )
        else:
            learning_model = SimpleMultiHeadModel(
                actor_net, 
                optim_option=OptimOption(optim_cls=Adam, optim_params=config.actor_optimizer)
            )
            agent_dict[agent_id] = PolicyGradient(agent_id, learning_model, config.reward_discount)

    return agent_dict
