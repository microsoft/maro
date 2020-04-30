# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


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
        

    def __call__(self, action_idx: int, warehouse_scope: int) -> int:
        '''
        Args:
            scope (ActionScope): Action actual available scope.
            action_index (int): Module output.
        '''

        return int(self._action_space[action_idx] * warehouse_scope)

    @property
    def action_space(self):
        return self._action_space
