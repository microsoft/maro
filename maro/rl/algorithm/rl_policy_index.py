# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .ac import ActorCritic, ActorCriticConfig, PolicyValueNetForDiscreteActionSpace
from .ddpg import DDPG, DDPGConfig, PolicyValueNetForContinuousActionSpace
from .dqn import DQN, DQNConfig, QNetForDiscreteActionSpace
from .pg import PolicyGradient, PolicyGradientConfig, PolicyNetForDiscreteActionSpace


RL_POLICY_INDEX = {
    "ac": ActorCritic,
    "dqn": DQN,
    "ddpg": DDPG,
    "pg": PolicyGradient
}

RL_POLICY_CONFIG_INDEX = {
    "ac": ActorCritic,
    "dqn": DQNConfig,
    "ddpg": DDPGConfig,
    "pg": PolicyGradientConfig
}

RL_POLICY_MODEL_INDEX = {
    "ac": PolicyValueNetForDiscreteActionSpace,
    "dqn": QNetForDiscreteActionSpace,
    "ddpg": PolicyValueNetForContinuousActionSpace,
    "pg": PolicyNetForDiscreteActionSpace
}


def get_rl_policy_cls(policy_type):
    if isinstance(policy_type, str):
        if policy_type not in RL_POLICY_INDEX:
            raise KeyError(f"A string policy_type must be one of {list(RL_POLICY_INDEX.keys())}.")
        return RL_POLICY_INDEX[policy_type]

    return policy_type


def get_rl_policy_config_cls(policy_config_type):
    if isinstance(policy_config_type, str):
        if policy_config_type not in RL_POLICY_CONFIG_INDEX:
            raise KeyError(f"A string policy_config_type must be one of {list(RL_POLICY_CONFIG_INDEX.keys())}.")
        return RL_POLICY_CONFIG_INDEX[policy_config_type]

    return policy_config_type


def get_rl_policy_model_cls(policy_model_type):
    if isinstance(policy_model_type, str):
        if policy_model_type not in RL_POLICY_MODEL_INDEX:
            raise KeyError(f"A string policy_model_type must be one of {list(RL_POLICY_MODEL_INDEX.keys())}.")
        return RL_POLICY_MODEL_INDEX[policy_model_type]

    return policy_model_type
