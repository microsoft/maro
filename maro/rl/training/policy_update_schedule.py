# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, namedtuple
from typing import List, Union

EpisodeBasedSchedule = namedtuple("EpisodeBasedSchedule", ["start", "interval"])
StepBasedSchedule = namedtuple("StepBasedSchedule", ["start_ep", "interval", "end_ep_update"])


class MultiPolicyUpdateSchedule:
    """
    """
    def __init__(self, schedule_option: Union[EpisodeBasedSchedule, StepBasedSchedule, dict] = -1):  
        self._pending_steps = defaultdict(list)
        self._pending_episodes = defaultdict(list)
        self._step_schedule_opt = {}
        self._episode_schedule_opt = {}

        if isinstance(schedule_option, dict):
            for policy_id, sch in schedule_option.items():
                if isinstance(sch, StepBasedSchedule):
                    self._step_schedule_opt[policy_id] = (sch.start_ep, sch.interval)
                    if sch.end_ep_update:
                        self._episode_schedule_opt[policy_id] = (sch.start_ep, 1)
                        self._pending_episodes[sch.start_ep].append(policy_id)
                    self._pending_steps[sch.interval].append(policy_id)
                elif isinstance(sch, EpisodeBasedSchedule):
                    self._episode_schedule_opt[policy_id] = (sch.start, sch.interval)
                    self._pending_episodes[sch.start].append(policy_id)
        else:
            if isinstance(schedule_option, StepBasedSchedule):
                self._step_schedule_opt["*"] = (schedule_option.start_ep, schedule_option.interval)
                self._pending_steps[schedule_option.interval].append("*")
                if schedule_option.end_ep_update:
                    self._episode_schedule_opt["*"] = (schedule_option.start_ep, 1)
                    self._pending_episodes[schedule_option.start_ep].append("*")
            else:
                self._episode_schedule_opt["*"] = (schedule_option.start, schedule_option.interval)
                self._pending_episodes[schedule_option.start].append("*")

        self._episode = None

    def pop_step(self, ep: int, step: int) -> List[str]:
        for policy_id in self._pending_steps[step]:
            next_step = step + self._step_schedule_opt[policy_id][1]
            # if the pending_steps is filled, skip the loop altogether
            if self._pending_steps[next_step]:
                break
            self._pending_steps[next_step].append(policy_id)

        return [
            policy_id for policy_id in self._pending_steps[step] if self._step_schedule_opt[policy_id][0] <= ep
        ]

    def pop_episode(self, ep: int) -> List[str]:
        for policy_id in self._pending_episodes[ep]:
            next_ep = ep + self._episode_schedule_opt[policy_id][1]
            self._pending_episodes[next_ep].append(policy_id)

        return self._pending_episodes[ep]