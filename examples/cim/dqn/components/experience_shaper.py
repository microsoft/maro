# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import ExperienceShaper


class TruncatedExperienceShaper(ExperienceShaper):
    def __init__(self, *, time_window: int, time_decay_factor: float, fulfillment_factor: float,
                 shortage_factor: float):
        super().__init__(reward_func=None)
        self._time_window = time_window
        self._time_decay_factor = time_decay_factor
        self._fulfillment_factor = fulfillment_factor
        self._shortage_factor = shortage_factor

    def __call__(self, trajectory, snapshot_list):
        experiences_by_agent = {}
        for i in range(len(trajectory) - 1):
            transition = trajectory[i]
            agent_id = transition["agent_id"]
            if agent_id not in experiences_by_agent:
                experiences_by_agent[agent_id] = defaultdict(list)
            experiences = experiences_by_agent[agent_id]
            experiences["state"].append(transition["state"])
            experiences["action"].append(transition["action"])
            experiences["reward"].append(self._compute_reward(transition["event"], snapshot_list))
            experiences["next_state"].append(trajectory[i + 1]["state"])

        return experiences_by_agent

    def _compute_reward(self, decision_event, snapshot_list):
        start_tick = decision_event.tick + 1
        end_tick = decision_event.tick + self._time_window
        ticks = list(range(start_tick, end_tick))

        # calculate tc reward
        future_fulfillment = snapshot_list["ports"][ticks::"fulfillment"]
        future_shortage = snapshot_list["ports"][ticks::"shortage"]
        decay_list = [self._time_decay_factor ** i for i in range(end_tick - start_tick)
                      for _ in range(future_fulfillment.shape[0] // (end_tick - start_tick))]

        tot_fulfillment = np.dot(future_fulfillment, decay_list)
        tot_shortage = np.dot(future_shortage, decay_list)

        return np.float(self._fulfillment_factor * tot_fulfillment - self._shortage_factor * tot_shortage)
