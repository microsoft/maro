# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .ac import ActorCritic, DiscreteACNet
from .ddpg import DDPG, ContinuousACNet
from .dqn import DQN, DiscreteQNet
from .pg import DiscretePolicyNet, PolicyGradient

POLICY_INDEX = {
    "ac": ActorCritic,
    "dqn": DQN,
    "ddpg": DDPG,
    "pg": PolicyGradient
}


MODEL_INDEX = {
    "ac": DiscreteACNet,
    "dqn": DiscreteQNet,
    "ddpg": ContinuousACNet,
    "pg": DiscretePolicyNet
}


def get_policy_cls(policy_type):
    if isinstance(policy_type, str):
        if policy_type not in POLICY_INDEX:
            raise KeyError(f"A string policy_type must be one of {list(POLICY_INDEX.keys())}.")
        return POLICY_INDEX[policy_type]

    return policy_type


def get_model_cls(model_type):
    if isinstance(model_type, str):
        if model_type not in MODEL_INDEX:
            raise KeyError(f"A string model_type must be one of {list(MODEL_INDEX.keys())}.")
        return MODEL_INDEX[model_type]

    return model_type
