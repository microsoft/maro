# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from abc import abstractmethod
from typing import Any

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
        explore_steps (int): Maximum number of steps to interpolate probability.
        start_explore_prob (float): Starting explore probability.
        end_explore_prob (float): Ending explore probability.
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
