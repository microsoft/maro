# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np


def eps_greedy(action: Union[int, np.ndarray], num_actions, *, epsilon: float, state=None):
    """epsilon-greedy exploration.

    Args:
        action (Union[int, np.ndarray]): Action(s) chosen greedily by the policy.
        num_actions (int): Number of possible actions.
        epsilon (float): The probability that a random action will be used 
        state: State information which might be needed as context to generate an exploratory action.
            In this simple epsilon-greedy scheme, it is not used. Defaults to None.
    """
    def get_exploration_action(action):
        return action if np.random.random() > epsilon else np.random.randint(num_actions)

    if isinstance(action, np.ndarray):
        return np.array([get_exploration_action(act) for act in action])
    else:
        return get_exploration_action(action)


def uniform_noise(
    action: Union[float, np.ndarray],
    *,
    low: Union[float, list, np.ndarray],
    high: Union[float, list, np.ndarray],
    min_action: Union[float, list, np.ndarray] = None,
    max_action: Union[float, list, np.ndarray] = None,
) -> Union[float, np.ndarray]:
    if min_action is None and max_action is None: 
        return action + np.random.uniform(low, high)
    else:
        return np.clip(action + np.random.uniform(low, high), min_action, max_action)


def gaussian_noise(
    action: Union[float, np.ndarray],
    *,
    mean: Union[float, list, np.ndarray] = .0,
    stddev: Union[float, list, np.ndarray] = 1.0,
    relative: bool = False,
    min_action: Union[float, list, np.ndarray] = None,
    max_action: Union[float, list, np.ndarray] = None,
) -> Union[float, np.ndarray]:
    noise = np.random.normal(loc=mean, scale=stddev)
    if min_action is None and max_action is None:
        return action + ((noise * action) if relative else noise)
    else:
        return np.clip(action + ((noise * action) if relative else noise), min_action, max_action)
