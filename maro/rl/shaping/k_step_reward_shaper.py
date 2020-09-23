# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable
from collections import deque

from maro.rl.common import ExperienceKey, ExperienceInfoKey, TransitionInfoKey
from .abstract_reward_shaper import AbstractRewardShaper


class KStepRewardShaper(AbstractRewardShaper):
    def __init__(self, reward_func: Callable, reward_decay: float, steps: int):
        super().__init__()
        self._reward_func = reward_func
        self._reward_decay = reward_decay
        self._steps = steps

    def _shape(self, snapshot_list):
        size = len(self._trajectory[ExperienceKey.STATE])
        reward_list, reward_sum = deque(), 0
        for i in range(size-2, -1, -1):
            target_idx = min(size-1, i+self._steps)
            self._trajectory[ExperienceKey.NEXT_STATE][i] = self._trajectory[ExperienceKey.STATE][target_idx]
            self._trajectory[ExperienceKey.NEXT_ACTION][i] = self._trajectory[ExperienceKey.ACTION][target_idx]
            reward_list.appendleft(self._reward_func(self._trajectory["extra"][i][TransitionInfoKey.METRICS]))
            if len(reward_list) > self._steps:
                reward_sum -= reward_list.pop() * self._reward_decay**(self._steps - 1)
            reward_sum = reward_sum * self._reward_decay + reward_list[0]
            self._trajectory[ExperienceKey.REWARD][i] = reward_sum
            self._trajectory["info"][i][ExperienceInfoKey.DISCOUNT] = self._reward_decay**(min(self._steps, size-1-i))
