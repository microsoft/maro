# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .ac import ActorCritic, ActorCriticConfig
from .ddpg import DDPG, DDPGConfig
from .dqn import DQN, DQNConfig
from .pg import PolicyGradient, PolicyGradientConfig


AGENT_CLS = {
    "ac": ActorCritic,
    "ddpg": DDPG,
    "dqn": DQN,
    "pg": PolicyGradient
}

AGENT_CONFIG = {
    "ac": ActorCriticConfig,
    "ddpg": DDPGConfig,
    "dqn": DQNConfig,
    "pg": PolicyGradientConfig
}
