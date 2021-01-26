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
        agent_ids = np.asarray(trajectory.get_by_key("agent_id"))
        states = np.asarray(trajectory.get_by_key("state"))
        actions = np.asarray(trajectory.get_by_key("action"))
        log_action_probabilities = np.asarray(trajectory.get_by_key("log_action_probability"))
        rewards = np.fromiter(
            map(self._compute_reward, trajectory.get_by_key("event"), [snapshot_list] * len(trajectory)),
            dtype=np.float32
        )
        return {agent_id: {
                    "state": states[agent_ids == agent_id],
                    "action": actions[agent_ids == agent_id],
                    "log_action_probability": log_action_probabilities[agent_ids == agent_id],
                    "reward": rewards[agent_ids == agent_id],
                }
                for agent_id in set(agent_ids)}

    def _compute_reward(self, decision_event, snapshot_list):
        start_tick = decision_event.tick + 1
        end_tick = decision_event.tick + self._time_window
        ticks = list(range(start_tick, end_tick))

        # calculate tc reward
        future_fulfillment = snapshot_list["ports"][ticks::"fulfillment"]
        future_shortage = snapshot_list["ports"][ticks::"shortage"]
        decay_list = [self._time_decay_factor ** i for i in range(end_tick - start_tick)
                      for _ in range(future_fulfillment.shape[0]//(end_tick-start_tick))]

        tot_fulfillment = np.dot(future_fulfillment, decay_list)
        tot_shortage = np.dot(future_shortage, decay_list)

        return np.float(self._fulfillment_factor * tot_fulfillment - self._shortage_factor * tot_shortage)
