# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os
from typing import Tuple

from maro.simulator.scenarios.ecr.common import ActionScope
from maro.utils import Logger, LogFormat

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
        zero_action_indexes = []
        for i, v in enumerate(self._action_space):
            if v == 0:
                zero_action_indexes.append(i)
        assert(len(zero_action_indexes) == 1)
        self._zero_action_index = zero_action_indexes[0]

    def __call__(self, scope: ActionScope, action_index: int, port_empty: float, vessel_remaining_space: float, early_discharge: int) -> int:
        '''
        Args:
            scope (ActionScope): Action actual available scope.
                        e.g. {'discharge': 0, 'load': 2000}
            action_index (int): Module output.
        '''
        assert(0 <= action_index < len(self._action_space))

        if action_index < self._zero_action_index:
            return max(round(self._action_space[action_index] * port_empty), -vessel_remaining_space)

        if action_index > self._zero_action_index:
            plan_action = self._action_space[action_index] * (scope.discharge + early_discharge) - early_discharge
            if plan_action > 0:
                return round(plan_action)
            else:
                return round(self._action_space[action_index] * scope.discharge) 

        return 0

    @property
    def action_space(self):
        return self._action_space

    @property
    def zero_action_index(self):
        return self._zero_action_index
