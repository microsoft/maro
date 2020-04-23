# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os

import numpy as np
import torch

from maro.simulator import Env
from maro.utils import Logger, LogFormat



class StateShaping():
    def __init__(self,
                 env: Env,
                 relative_tick_list: [int],
                 port_downstream_max_number: int,
                 port_attribute_list: [str],
                 vessel_attribute_list: [str]
                 ):
        self._env = env
        self._relative_tick_list = relative_tick_list
        self._port_downstream_max_number = port_downstream_max_number
        self._port_attribute_list = port_attribute_list
        self._vessel_attribute_list = vessel_attribute_list
        self._dim = (len(self._relative_tick_list) + 1) * \
            (self._port_downstream_max_number + 1) * \
            len(self._port_attribute_list) + len(self._vessel_attribute_list)

    def __call__(self, cur_tick: int, cur_port_idx: int, cur_vessel_idx: int):
        ticks = [cur_tick] + [cur_tick + rt for rt in self._relative_tick_list]
        future_port_slot_idx_list = [i for i in range(
            self._port_downstream_max_number)]
        future_port_idx_list = self._env.snapshot_list.dynamic_nodes[cur_tick: cur_vessel_idx: 'future_stop_list']

        port_features = self._env.snapshot_list.static_nodes[ticks: [cur_port_idx] + list(future_port_idx_list): self._port_attribute_list]

        vessel_features = self._env.snapshot_list.dynamic_nodes[cur_tick: cur_vessel_idx: self._vessel_attribute_list]

        res = np.concatenate((port_features, vessel_features))

        return res

    @property
    def dim(self):
        return self._dim
