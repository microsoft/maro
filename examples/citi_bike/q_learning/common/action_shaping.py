# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os
from typing import Tuple

from maro.simulator.scenarios.ecr.common import ActionScope
from maro.utils import Logger, LogFormat
from maro.simulator.scenarios.bike.common import Action, DecisionEvent

class DiscreteActionShaping():
    def __init__(self, action_space: [float]):
        '''
        Init action shaping.
        Args:
            action_space ([float]): Discrete action space, which must include a zero
                                    action, can be symmetry or asymmetric.
                                    i.e. [-1.0, -0.9, ... , 0 , ... , 0.9, 1.0]
                                         [-1.0, -0.5, 0, ... , 0.9, 1.0]
        '''
        self._action_space = action_space
        

    def __call__(self, action_idx: int, station_scope: int, neighbor_scope: int) -> int:
        '''
        Args:
            scope (ActionScope): Action actual available scope.
                        e.g. {'discharge': 0, 'load': 2000}
            action_index (int): Module output.
        '''

        return min(int(self._action_space[action_idx] * station_scope), neighbor_scope)

    @property
    def action_space(self):
        return self._action_space


class ListActionShaping():
    def __init__(self):
        '''
        Init action list.
        Args:
            action_space ([float]): Discrete action space, which must include a zero
                                    action, can be symmetry or asymmetric.
                                    i.e. [-1.0, -0.9, ... , 0 , ... , 0.9, 1.0]
                                         [-1.0, -0.5, 0, ... , 0.9, 1.0]
        '''
        self._action_scope_list = {}

    def append_action_scope(self, action_scope):
        for key,value in action_scope.items():
            if key not in self._action_scope_list:
                self._action_scope_list[key] = value
    
    def clear_scope_cache(self):
        self._action_scope_list = {}

    def __call__(self, action_list: [Action]) -> list:
        '''
        Args:
            scope (ActionScope): Action actual available scope.
                        e.g. {'discharge': 0, 'load': 2000}
            action_index (int): Module output.
        '''
        sum_to_cell = {}

        for action in action_list:
            if action.to_cell in sum_to_cell.keys():
                sum_to_cell[action.to_cell] += action.number
            else:
                sum_to_cell[action.to_cell] = action.number

        for action in action_list:
            if sum_to_cell[action.to_cell] > self._action_scope_list[action.to_cell]:
                action.number = int(action.number* self._action_scope_list[action.to_cell]//sum_to_cell[action.to_cell])

        self.clear_scope_cache()
        return action_list
