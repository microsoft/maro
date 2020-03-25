# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os

import numpy as np
import torch

from maro.simulator import Env
from maro.simulator.graph import ResourceNodeType
from maro.utils import Logger, LogFormat



class StateShaping():
    def __init__(self,
                 env: Env,
                 relative_tick_list: [int],
                 station_attribute_list: [str],
                 neighbor_number: int,
                 ):
        self._env = env
        self._relative_tick_list = relative_tick_list
        self._station_attribute_list = station_attribute_list
        self._dim = (len(self._relative_tick_list) + 1) * \
            (neighbor_number + 1) * \
            len(self._station_attribute_list)
            
    def __call__(self, cur_tick: int, cur_station_idx: int):
        ticks = [cur_tick] + [cur_tick + rt for rt in self._relative_tick_list]
        # To do: obtain neighbor id from snapshot list
        # cur_neighbor_idx_list = self._env.snapshot_list.static_nodes[ticks: [cur_station_idx]: ('neightbors', 0)]
        cur_neighbor_idx_list = [0]*6
        station_features = self._env.snapshot_list.static_nodes[ticks: [cur_station_idx] + list(cur_neighbor_idx_list): (self._station_attribute_list, 0)]

        return station_features

    @property
    def dim(self):
        return self._dim
