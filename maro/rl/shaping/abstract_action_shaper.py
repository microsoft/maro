# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbstractActionShaper(ABC):
    """
    An action shaper is used to convert the output of an underlying algorithm's choose_action() method to    \n
    an Action object which can be executed by the env's step() method.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, model_action, decision_event, snapshot_list):
        return NotImplementedError
