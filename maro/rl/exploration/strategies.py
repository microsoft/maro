# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np


def epsilon_greedy(
    state: np.ndarray,
    action: np.ndarray,
    num_actions: int,
    *,
    epsilon: float,
) -> np.ndarray:
    """Epsilon-greedy exploration.

    Args:
        state (np.ndarray): State(s) based on which ``action`` is chosen. This is not used by the vanilla
            eps-greedy exploration and is put here to conform to the function signature required for the exploration
            strategy parameter for ``DQN``.
        action (np.ndarray): Action(s) chosen greedily by the policy.
        num_actions (int): Number of possible actions.
        epsilon (float): The probability that a random action will be selected.

    Returns:
        Exploratory actions.
    """
    return np.array([act if np.random.random() > epsilon else np.random.randint(num_actions) for act in action])


def uniform_noise(
    state: np.ndarray,
    action: np.ndarray,
    min_action: Union[float, list, np.ndarray] = None,
    max_action: Union[float, list, np.ndarray] = None,
    *,
    low: Union[float, list, np.ndarray],
    high: Union[float, list, np.ndarray],
) -> Union[float, np.ndarray]:
    """Apply a uniform noise to a continuous multidimensional action.

    Args:
        state (np.ndarray): State(s) based on which ``action`` is chosen. This is not used by the gaussian noise
            exploration scheme and is put here to conform to the function signature for the exploration in continuous
            action spaces.
        action (np.ndarray): Action(s) chosen greedily by the policy.
        min_action (Union[float, list, np.ndarray], default=None): Lower bound for the multidimensional action space.
        max_action (Union[float, list, np.ndarray], default=None): Upper bound for the multidimensional action space.
        low (Union[float, list, np.ndarray]): Lower bound for the noise range.
        high (Union[float, list, np.ndarray]): Upper bound for the noise range.

    Returns:
        Exploration actions with added noise.
    """
    if min_action is None and max_action is None:
        return action + np.random.uniform(low, high, size=action.shape)
    else:
        return np.clip(action + np.random.uniform(low, high, size=action.shape), min_action, max_action)


def gaussian_noise(
    state: np.ndarray,
    action: np.ndarray,
    min_action: Union[float, list, np.ndarray] = None,
    max_action: Union[float, list, np.ndarray] = None,
    *,
    mean: Union[float, list, np.ndarray] = 0.0,
    stddev: Union[float, list, np.ndarray] = 1.0,
    relative: bool = False,
) -> Union[float, np.ndarray]:
    """Apply a gaussian noise to a continuous multidimensional action.

    Args:
        state (np.ndarray): State(s) based on which ``action`` is chosen. This is not used by the gaussian noise
            exploration scheme and is put here to conform to the function signature for the exploration in continuous
            action spaces.
        action (np.ndarray): Action(s) chosen greedily by the policy.
        min_action (Union[float, list, np.ndarray], default=None): Lower bound for the multidimensional action space.
        max_action (Union[float, list, np.ndarray], default=None): Upper bound for the multidimensional action space.
        mean (Union[float, list, np.ndarray], default=0.0): Gaussian noise mean.
        stddev (Union[float, list, np.ndarray], default=1.0): Standard deviation for the Gaussian noise.
        relative (bool, default=False): If True, the generated noise is treated as a relative measure and will
            be multiplied by the action itself before being added to the action.

    Returns:
        Exploration actions with added noise (a numpy ndarray).
    """
    noise = np.random.normal(loc=mean, scale=stddev, size=action.shape)
    if min_action is None and max_action is None:
        return action + ((noise * action) if relative else noise)
    else:
        return np.clip(action + ((noise * action) if relative else noise), min_action, max_action)
