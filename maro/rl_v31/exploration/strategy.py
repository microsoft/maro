# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from maro.rl_v31.utils import to_torch


class ExploreStrategy:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_action(
        self,
        obs: Any,
        action: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            obs (Any): Observation(s) based on which ``action`` is chosen. This is not used by the vanilla
                eps-greedy exploration and is put here to conform to the function signature required for the exploration
                strategy parameter for ``DQN``.
            action (torch.Tensor): Action(s) chosen greedily by the policy.

        Returns:
            Exploratory actions (torch.Tensor).
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
        super().__init__()

        assert 0.0 <= epsilon <= 1.0

        self._num_actions = num_actions
        self._eps = epsilon

    def get_action(
        self,
        obs: Any,
        action: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ret = np.array(
            [act if np.random.random() > self._eps else np.random.randint(self._num_actions) for act in action],
        )
        return to_torch(ret)


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
        super().__init__()

        self._call_count = 0

        self._num_actions = num_actions
        self._explore_steps = explore_steps
        self._start_explore_prob = start_explore_prob
        self._end_explore_prob = end_explore_prob

    def get_action(
        self,
        obs: Any,
        action: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ratio = min(self._call_count / self._explore_steps, 1.0)
        epsilon = self._start_explore_prob + (self._end_explore_prob - self._start_explore_prob) * ratio
        explore_flag = np.random.random() < epsilon
        action = np.array([np.random.randint(self._num_actions) if explore_flag else act for act in action])

        self._call_count += 1
        return to_torch(action)


class GaussianNoise(ExploreStrategy):
    def __init__(self, noise_scale: float, action_limit: float) -> None:
        super().__init__()

        self._noise_scale = noise_scale
        self._action_limit = action_limit

    def get_action(
        self,
        obs: Any,
        action: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        action += torch.randn(action.shape[1]) * self._noise_scale
        action = torch.clamp(action, -self._action_limit, self._action_limit)
        return action
