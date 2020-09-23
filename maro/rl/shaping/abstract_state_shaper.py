# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbstractStateShaper(ABC):
    """
    A state shaper is used to convert a decision event and snapshot list to a state vector as input to value or
    policy models by extracting relevant temporal and spatial information.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, decision_event, snapshot_list):
        return NotImplementedError
