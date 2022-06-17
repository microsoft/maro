# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


def or_epsilon_greedy(
    state: np.ndarray,
    action: np.ndarray,
    num_actions: int,
    or_actions: int,
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
    buffer = 2
    rnd_actions_upper = np.where(or_actions + buffer > num_actions, num_actions, or_actions + buffer)
    rnd_actions_lower = np.where(or_actions - buffer < 0, 0, or_actions - buffer)
    rnd_actions = np.random.randint(rnd_actions_lower, rnd_actions_upper)  # half open
    return np.array([act if np.random.random() > epsilon else rnd_act for act, rnd_act in zip(action, rnd_actions)])
