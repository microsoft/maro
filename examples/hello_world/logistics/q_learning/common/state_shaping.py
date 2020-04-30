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
                 warehouse_attribute_list: [str]):
        self._env = env
        self._relative_tick_list = relative_tick_list
        self._warehouse_attribute_list = warehouse_attribute_list
        self._dim = (len(self._relative_tick_list) + 1) * \
            len(self._warehouse_attribute_list)

    def __call__(self, cur_tick: int, cur_warehouse_idx: int):
        ticks = [cur_tick] + [cur_tick + rt for rt in self._relative_tick_list]
        warehouse_features = self._env.snapshot_list.static_nodes[ticks: [cur_warehouse_idx]: (self._warehouse_attribute_list, 0)]
        return warehouse_features

    @property
    def dim(self):
        return self._dim
