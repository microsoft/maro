# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle

import numpy as np

from maro.rl import AbstractRewardShaper, ExperienceKey, ExperienceInfoKey, TransitionInfoKey


class ECRRewardShaper(AbstractRewardShaper):
    def __init__(self, *, agent_id_list, time_window: int, time_decay_factor: float,
                 fulfillment_factor: float, shortage_factor: float):
        super().__init__()
        self._agent_id_list = agent_id_list
        self._time_window = time_window
        self._time_decay_factor = time_decay_factor
        self._fulfillment_factor = fulfillment_factor
        self._shortage_factor = shortage_factor

    def _shape(self, snapshot_list):
        for i in range(len(self._trajectory[ExperienceKey.STATE])-1):
            metrics = self._trajectory["extra"][i][TransitionInfoKey.METRICS]
            event = pickle.loads(self._trajectory["extra"][i][TransitionInfoKey.EVENT])
            self._trajectory[ExperienceKey.REWARD][i] = self._compute_reward(metrics, event, snapshot_list)
            self._trajectory[ExperienceKey.NEXT_STATE][i] = self._trajectory[ExperienceKey.STATE][i+1]
            self._trajectory[ExperienceKey.NEXT_ACTION][i] = self._trajectory[ExperienceKey.ACTION][i+1]
            self._trajectory["info"][i][ExperienceInfoKey.DISCOUNT] = .0

    def _compute_reward(self, metrics, decision_event, snapshot_list):
        start_tick = decision_event.tick + 1
        end_tick = decision_event.tick + self._time_window
        ticks = list(range(start_tick, end_tick))

        # calculate tc reward
        decay_list = [self._time_decay_factor ** i for i in range(end_tick - start_tick)
                      for _ in range(len(self._agent_id_list))]

        tot_fulfillment = np.dot(snapshot_list["ports"][ticks::"fulfillment"], decay_list)
        tot_shortage = np.dot(snapshot_list["ports"][ticks::"shortage"], decay_list)

        return np.float(self._fulfillment_factor * tot_fulfillment - self._shortage_factor * tot_shortage)
