# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import Shaper


class CIMExperienceShaper(Shaper):
    def __init__(self, *, time_window: int, time_decay_factor: float, fulfillment_factor: float,
                 shortage_factor: float):
        super().__init__(reward_func=None)
        self._time_window = time_window
        self._time_decay_factor = time_decay_factor
        self._fulfillment_factor = fulfillment_factor
        self._shortage_factor = shortage_factor
        self._trajectory = {key: [] for key in ["state", "action", "agent_id", "event", "log_action_prob"]}

    def __call__(self, snapshot_list):
        agent_ids = self._trajectory["agent_id"]
        events = self._trajectory["event"]
        states = self._trajectory["state"]
        actions = self._trajectory["action"]
        log_action_prob = self._trajectory["log_action_prob"]
        
        exp_by_agent = defaultdict(lambda: defaultdict(list))
        for i in range(len(states)):
            exp_by_agent[agent_ids[i]]["state"].append(states[i])
            exp_by_agent[agent_ids[i]]["action"].append(actions[i])
            exp_by_agent[agent_ids[i]]["log_action_prob"].append(log_action_prob[i])
            exp_by_agent[agent_ids[i]]["reward"].append(self._compute_reward(events[i], snapshot_list))
            
        for agent_id in exp_by_agent:
            for key, vals in exp_by_agent[agent_id].items():
                exp_by_agent[agent_id][key] = np.asarray(vals, dtype=np.float32 if key == "reward" else None)
        
        return exp_by_agent

    def record(self, transition: dict):
        for key, val in transition.items():
            self._trajectory[key].append(val)
    
    def reset(self):
        self._trajectory = {key: [] for key in ["state", "action", "agent_id", "event", "log_action_prob"]}
    
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
