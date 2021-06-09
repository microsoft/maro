# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from abc import ABC, abstractmethod
from typing import List

from .experience_manager import ExperienceManager, ExperienceSet


class AbsSampler(ABC):
    """Sampler class.

    Args:
        experience_manager (ExperienceManager): experience manager the sampler is associated with.
    """
    def __init__(self, experience_manager: ExperienceManager):
        super().__init__()
        self.experience_manager = experience_manager

    @abstractmethod
    def get(self) -> ExperienceSet:
        """Sampling logic is defined here."""
        raise NotImplementedError

    def on_put(self, experience_set: ExperienceSet, indexes: List[int]):
        """Callback to be executed after calling experience_manager.put()."""
        pass


class PrioritizedSampler(AbsSampler):
    """Sampler for Prioritized Experience Replay (PER).

    References:
        https://arxiv.org/pdf/1511.05952.pdf
        https://github.com/rlcode/per

    Args:
        experience_manager (ExperienceManager): experience manager the sampler is associated with.
        batch_size (int): mini-batch size. Defaults to 32.
        alpha (float): Prioritization strength. Sampling probabilities are calculated according to
            P = p_i^alpha / sum(p_k^alpha). Defaults to 0.6.
        beta (float): Bias annealing strength using weighted importance sampling (IS) techniques.
            IS weights are calculated according to (N * P)^(-beta), where P is the sampling probability.
            This value of ``beta`` should not exceed 1.0, which corresponds to full annealing. Defaults to 0.4.
        beta_step (float): The amount ``beta`` is incremented by after each get() call until it reaches 1.0.
            Defaults to 0.001.
    """
    def __init__(
        self,
        experience_manager: ExperienceManager,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_step: float = 0.001
    ):
        if beta > 1.0:
            raise ValueError("beta should be between 0.0 and 1.0")
        super().__init__(experience_manager)
        self._sum_tree = np.zeros(2 * self.experience_manager.capacity - 1)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.eps = 1e-7
        self._max_priority = 1e8

    def total(self):
        """Return the sum of priorities over all experiences."""
        return self._sum_tree[0]

    def on_put(self, experience_set: ExperienceSet, indexes: List[int]):
        """Set the priorities of newly added experiences to the maximum value."""
        self.update(indexes, [self._max_priority] * len(indexes))

    def update(self, indexes, td_errors):
        """Update priority values at given indexes."""
        for idx, err in zip(indexes, td_errors):
            priority = self._get_priority(err)
            tree_idx = idx + self.experience_manager.capacity - 1
            delta = priority - self._sum_tree[tree_idx]
            self._sum_tree[tree_idx] = priority
            self._update(tree_idx, delta)

    def get(self):
        """Priority-based sampling."""
        indexes, priorities = [], []
        segment_len = self.total() / self.batch_size
        for i in range(self.batch_size):
            low, high = segment_len * i, segment_len * (i + 1)
            sampled_val = np.random.uniform(low=low, high=high)
            idx = self._get(0, sampled_val)
            data_idx = idx - self.experience_manager.capacity + 1
            indexes.append(data_idx)
            priorities.append(self._sum_tree[idx])

        self.beta = min(1., self.beta + self.beta_step)
        sampling_probabilities = priorities / self.total()
        is_weights = np.power(self.experience_manager.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return ExperienceSet(
            states=[self.experience_manager.data["states"][idx] for idx in indexes],
            actions=[self.experience_manager.data["actions"][idx] for idx in indexes],
            rewards=[self.experience_manager.data["rewards"][idx] for idx in indexes],
            next_states=[self.experience_manager.data["next_states"][idx] for idx in indexes],
            info=[{"index": idx, "is_weight": wt} for idx, wt in zip(indexes, is_weights)]
        )

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def _update(self, idx, delta):
        """Propagate priority change all the way to the root node."""
        parent = (idx - 1) // 2
        self._sum_tree[parent] += delta
        if parent != 0:
            self._update(parent, delta)

    def _get(self, idx, sampled_val):
        """Get a leaf node according to a randomly sampled value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._sum_tree):
            return idx

        if sampled_val <= self._sum_tree[left]:
            return self._get(left, sampled_val)
        else:
            return self._get(right, sampled_val - self._sum_tree[left])
