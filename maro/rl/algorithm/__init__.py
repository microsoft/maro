# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .abs_algorithm import AbsAlgorithm
from .ac import ActorCritic, ActorCriticConfig
from .ddpg import DDPG, DDPGConfig
from .dqn import DQN, DQNConfig
from .pg import PolicyGradient, PolicyGradientConfig
from .torch_loss_cls_index import TORCH_LOSS_CLS

__all__ = [
    "AbsAlgorithm",
    "ActorCritic", "ActorCriticConfig",
    "DDPG", "DDPGConfig",
    "DQN", "DQNConfig",
    "MultiAgentWrapper",
    "PolicyGradient", "PolicyGradientConfig",
    "TORCH_LOSS_CLS"
]
