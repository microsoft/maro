# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import heapq
from collections import namedtuple
from typing import List, Union

EpisodeBasedSchedule = namedtuple("EpisodeBasedSchedule", ["start", "interval"])
StepBasedSchedule = namedtuple("StepBasedSchedule", ["start_ep", "interval", "end_ep_update"])


class MultiPolicyUpdateSchedule:
    """
    """
    def __init__(self, schedule_option: Union[EpisodeBasedSchedule, StepBasedSchedule, dict] = -1):  
        self._pending_steps = []
        self._pending_episodes = []
        if isinstance(schedule_option, dict):
            self._step_schedule_opt = {}
            self._episode_schedule_opt = {}
            for policy_id, sch in schedule_option.items():
                if isinstance(sch, StepBasedSchedule):
                    self._step_schedule_opt[policy_id] = (sch.start_ep, sch.interval)
                    if sch.end_ep_update:
                        self._episode_schedule_opt[policy_id] = (sch.start_ep, 1)
                elif isinstance(sch, EpisodeBasedSchedule):
                    self._episode_schedule_opt[policy_id] = (sch.start, sch.interval)

            for policy_id, (start, _) in self._episode_schedule_opt.items():
                heapq.heappush(self._pending_episodes, (start, policy_id))
        else:
            self._episode_schedule_opt = None
            self._step_schedule_opt = None
            if isinstance(schedule_option, EpisodeBasedSchedule):
                self._episode_schedule_opt = (schedule_option.start, schedule_option.interval)
            else:
                self._step_schedule_opt = (schedule_option.start_ep, schedule_option.interval)
            
            heapq.heappush(self._pending_episodes, (start, "*"))

        self._in_episode = False

    def enter(self, ep: int):
        self._in_episode = True
        for policy_id, (start_ep, interval) in self._step_schedule_opt.items():
            if ep >= start_ep:
                heapq.heappush(self._pending_steps, (interval, policy_id))

    def exit(self):
        self._in_episode = False
        self._pending_steps.clear()

    def pop(self, index: int) -> List[str]:
        policy_ids = []
        pending = self._pending_steps if self._in_episode else self._pending_episodes
        schedule_opt = self._step_schedule_opt if self._in_episode else self._episode_schedule_opt
        while pending and pending[0][0] == index:
            _, policy_id = heapq.heappop(pending)
            policy_ids.append(policy_id)
            next_index = index + schedule_opt[policy_id][1]
            heapq.heappush(pending, (next_index, policy_id))

        return policy_ids
