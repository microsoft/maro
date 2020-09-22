# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from .abs_shaper import AbsShaper


class StateShaper(AbsShaper):
    """
    A state shaper is used to convert a decision event and snapshot list to a state vector as input to value or
    policy models by extracting relevant temporal and spatial information.
    """
    @abstractmethod
    def __call__(self, decision_event, snapshot_list):
        pass

    def reset(self):
        pass
