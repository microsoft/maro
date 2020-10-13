# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np

from .experience_shaper import ExperienceShaper
from maro.rl.utils.trajectory_utils import get_k_step_discounted_sums


class KStepExperienceKeys(Enum):
    STATE = "state"
    ACTION = "action"
    REWARD = "reward"
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
        length = len(trajectory)
        agent_ids = np.asarray(trajectory.get_by_key["agent_id"])
        states = np.asarray(trajectory.get_by_key["state"])
        actions = np.asarray(trajectory.get_by_key["action"])
        reward_array = np.fromiter(map(self._reward_func, trajectory.get_by_key("metrics")), dtype=np.float32)
        reward_sums = get_k_step_discounted_sums(reward_array, self._reward_decay, k=self._steps)[:-1]
        discounts = np.array([self._reward_decay ** min(self._steps, length-i-1) for i in range(length-1)])
        next_states = np.pad(states[self._steps:], (0, length-self._steps-1), mode="edge")
        next_actions = np.pad(actions[self._steps:], (0, length-self._steps-1), mode="edge")

        states, actions = states[:-1], actions[:-1]

        if self._is_per_agent:
            return {agent_id: {KStepExperienceKeys.STATE.value: states[agent_ids == agent_id],
                               KStepExperienceKeys.ACTION.value: actions[agent_ids == agent_id],
                               KStepExperienceKeys.REWARD.value: reward_sums[agent_ids == agent_id],
                               KStepExperienceKeys.NEXT_STATE.value: next_states[agent_ids == agent_id],
                               KStepExperienceKeys.NEXT_ACTION.value: next_actions[agent_ids == agent_id],
                               KStepExperienceKeys.DISCOUNT.value: discounts[agent_ids == agent_id]}
                    for agent_id in set(agent_ids)}
        else:
            return {KStepExperienceKeys.STATE.value: states,
                    KStepExperienceKeys.ACTION.value: actions,
                    KStepExperienceKeys.REWARD.value: reward_sums,
                    KStepExperienceKeys.NEXT_STATE.value: next_states,
                    KStepExperienceKeys.NEXT_ACTION.value: next_actions,
                    KStepExperienceKeys.DISCOUNT.value: discounts}
