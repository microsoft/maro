# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from abc import abstractmethod
from typing import Any, Union

import numpy as np


class ExploreStrategy:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Args:
            state (np.ndarray): State(s) based on which ``action`` is chosen. This is not used by the vanilla
                eps-greedy exploration and is put here to conform to the function signature required for the exploration
                strategy parameter for ``DQN``.
            action (np.ndarray): Action(s) chosen greedily by the policy.

        Returns:
            Exploratory actions.
        """
        raise NotImplementedError


class EpsilonGreedy(ExploreStrategy):
    """Epsilon-greedy exploration. Returns uniformly random action with probability `epsilon` or returns original
    action with probability `1.0 - epsilon`.

    Args:
        num_actions (int): Number of possible actions.
        epsilon (float): The probability that a random action will be selected.
    """

    def __init__(self, num_actions: int, epsilon: float) -> None:
        super(EpsilonGreedy, self).__init__()

        assert 0.0 <= epsilon <= 1.0

        self._num_actions = num_actions
        self._eps = epsilon

    def get_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.array(
            [act if np.random.random() > self._eps else np.random.randint(self._num_actions) for act in action],
        )


class LinearExploration(ExploreStrategy):
    """Epsilon greedy which the probability `epsilon` is linearly interpolated between `start_explore_prob` and
    `end_explore_prob` over `explore_steps`. After this many timesteps pass, `epsilon` is fixed to `end_explore_prob`.

    Args:
        num_actions (int): Number of possible actions.
        explore_steps (int)
        start_explore_prob (float)
        end_explore_prob (float)
    """

    def __init__(
        self,
        num_actions: int,
        explore_steps: int,
        start_explore_prob: float,
        end_explore_prob: float,
    ) -> None:
        super(LinearExploration, self).__init__()

        self._call_count = 0

        self._num_actions = num_actions
        self._explore_steps = explore_steps
        self._start_explore_prob = start_explore_prob
        self._end_explore_prob = end_explore_prob

    def get_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        ratio = min(self._call_count / self._explore_steps, 1.0)
        epsilon = self._start_explore_prob + (self._end_explore_prob - self._start_explore_prob) * ratio
        explore_flag = np.random.random() < epsilon
        action = np.array([np.random.randint(self._num_actions) if explore_flag else act for act in action])

        self._call_count += 1
        return action


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
