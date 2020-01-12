# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os

import numpy as np
import torch

from maro.simulator import Env

class StateShaping():
    def __init__(self,
                 env: Env,
                 relative_tick_list: [int],
                 port_downstream_max_number: int,
                 port_attribute_list: [str],
                 vessel_attribute_list: [str],
                 use_port_index: bool = False,
                 use_vessel_index: bool = False
                 ):
        self._env = env
        self._relative_tick_list = relative_tick_list
        self._port_downstream_max_number = port_downstream_max_number
        self._port_attribute_list = port_attribute_list
        self._vessel_attribute_list = vessel_attribute_list

        self._port_index_dict = {}
        self._vessel_index_dict = {}
        total_port_num = len(self._env.node_name_mapping['static'].keys())
        total_vessel_num = len(self._env.node_name_mapping['dynamic'].keys())
        for port_idx in self._env.node_name_mapping['static'].keys():
            self._port_index_dict[port_idx] = np.eye(total_port_num, dtype=np.float32)[port_idx] if use_port_index else np.array([], dtype=np.float32)
        for vessel_idx in self._env.node_name_mapping['dynamic'].keys():
            self._vessel_index_dict[vessel_idx] = np.eye(total_vessel_num, dtype=np.float32)[vessel_idx] if use_vessel_index else np.array([], dtype=np.float32)
        # TODO: add route index directly

        self._dim = (len(self._relative_tick_list) + 1) * (self._port_downstream_max_number + 1) * len(self._port_attribute_list) \
            + len(self._vessel_attribute_list) \
            + (len(self._env.agent_idx_list) if use_port_index else 0) \
            + (len(self._env.node_name_mapping['dynamic'].keys()) if use_vessel_index else 0)

    def __call__(self, cur_tick: int, cur_port_idx: int, cur_vessel_idx: int):
        ticks = [cur_tick] + [cur_tick + rt for rt in self._relative_tick_list]
        future_port_slot_idx_list = [i for i in range(
            self._port_downstream_max_number)]
        future_port_idx_list = self._env.snapshot_list.dynamic_nodes[cur_tick: cur_vessel_idx: ('future_stop_list', future_port_slot_idx_list)]

        port_features = self._env.snapshot_list.static_nodes[ticks: [cur_port_idx] + list(future_port_idx_list): (self._port_attribute_list, 0)]

        vessel_features = self._env.snapshot_list.dynamic_nodes[cur_tick: cur_vessel_idx: (self._vessel_attribute_list, 0)]

        res = np.concatenate((port_features,
                              vessel_features,
                              self._port_index_dict[cur_port_idx],
                              self._vessel_index_dict[cur_vessel_idx]
                              ))

        return res

    @property
    def dim(self):
        return self._dim
