# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .ac import ActorCritic, DiscreteACNet
from .ddpg import DDPG, ContinuousACNet
from .dqn import DQN, DiscreteQNet
from .pg import DiscretePolicyNet, PolicyGradient

ALGORITHM_INDEX = {
    "ac": ActorCritic,
    "dqn": DQN,
    "ddpg": DDPG,
    "pg": PolicyGradient
}


ALGORITHM_MODEL_INDEX = {
    "ac": DiscreteACNet,
    "dqn": DiscreteQNet,
    "ddpg": ContinuousACNet,
    "pg": DiscretePolicyNet
}


def get_algorithm_cls(algorithm_type):
    if isinstance(algorithm_type, str):
        if algorithm_type not in ALGORITHM_INDEX:
            raise KeyError(f"A string algorithm_type must be one of {list(ALGORITHM_INDEX.keys())}.")
        return ALGORITHM_INDEX[algorithm_type]

    return algorithm_type


def get_algorithm_model_cls(algorithm_model_type):
    if isinstance(algorithm_model_type, str):
        if algorithm_model_type not in ALGORITHM_MODEL_INDEX:
            raise KeyError(f"A string algorithm_model_type must be one of {list(ALGORITHM_MODEL_INDEX.keys())}.")
        return ALGORITHM_MODEL_INDEX[algorithm_model_type]

    return algorithm_model_type
