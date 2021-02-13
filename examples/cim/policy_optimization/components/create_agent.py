# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import Adam, RMSprop

from maro.rl import (
    ActorCritic, ActorCriticConfig, FullyConnectedBlock, OptimOption, PolicyGradient, SimpleMultiHeadModel
)


def create_po_agent(config):
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
        ac_model = SimpleMultiHeadModel(
            {"actor": actor_net, "critic": critic_net}, 
            optim_option={
                "actor": OptimOption(optim_cls=Adam, optim_params=config.actor_optimizer),
                "critic": OptimOption(optim_cls=RMSprop, optim_params=config.critic_optimizer)
            }
        )
        return ActorCritic(ac_model, ActorCriticConfig(critic_loss_func=nn.SmoothL1Loss(), **hyper_params))
    else:
        policy_model = SimpleMultiHeadModel(
            actor_net, 
            optim_option=OptimOption(optim_cls=Adam, optim_params=config.actor_optimizer)
        )
        return PolicyGradient(policy_model, config.reward_discount)
