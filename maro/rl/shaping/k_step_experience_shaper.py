# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, deque
from enum import Enum
from typing import Callable

from .experience_shaper import ExperienceShaper


class KStepExperienceKeys(Enum):
    STATE = "state"
    ACTION = "action"
    REWARD = "reward"
    RETURN = "return"
    NEXT_STATE = "next_state"
    NEXT_ACTION = "next_action"
    DISCOUNT = "discount"


class KStepExperienceShaper(ExperienceShaper):
    """Experience shaper to generate K-step and full returns for each transition along a trajectory.

    Args:
        reward_func (Callable): a function used to compute immediate rewards from metrics given by the env.
        reward_decay (float): decay factor used to evaluate multi-step returns.
        steps (int): number of time steps used in computing returns
        is_per_agent (bool): if True, the generated experiences will be bucketed by agent ID.
    """
    def __init__(self, reward_func: Callable, reward_decay: float, steps: int, is_per_agent: bool = True):
        super().__init__(reward_func)
        self._reward_decay = reward_decay
        self._steps = steps
        self._is_per_agent = is_per_agent

    def __call__(self, trajectory, snapshot_list):
        experiences = defaultdict(lambda: defaultdict(deque)) if self._is_per_agent else defaultdict(deque)
        reward_list = deque()
        full_return = partial_return = 0
        for i in range(len(trajectory) - 2, -1, -1):
            transition = trajectory[i]
            next_transition = trajectory[min(len(trajectory) - 1, i + self._steps)]
            reward_list.appendleft(self._reward_func(trajectory[i]["metrics"]))
            # compute the full return
            full_return = full_return * self._reward_decay + reward_list[0]
            # compute the partial return
            partial_return = partial_return * self._reward_decay + reward_list[0]
            if len(reward_list) > self._steps:
                partial_return -= reward_list.pop() * self._reward_decay ** (self._steps - 1)
            agent_exp = experiences[transition["agent_id"]] if self._is_per_agent else experiences
            agent_exp[KStepExperienceKeys.STATE.value].appendleft(transition["state"])
            agent_exp[KStepExperienceKeys.ACTION.value].appendleft(transition["action"])
            agent_exp[KStepExperienceKeys.REWARD.value].appendleft(partial_return)
            agent_exp[KStepExperienceKeys.RETURN.value].appendleft(full_return)
            agent_exp[KStepExperienceKeys.NEXT_STATE.value].appendleft(next_transition["state"])
            agent_exp[KStepExperienceKeys.NEXT_ACTION.value].appendleft(next_transition["action"])
            agent_exp[KStepExperienceKeys.DISCOUNT.value].appendleft(
                self._reward_decay ** (min(self._steps, len(trajectory) - 1 - i))
            )

        return dict(experiences)
