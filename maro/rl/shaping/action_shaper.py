# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from .abs_shaper import AbsShaper


class ActionShaper(AbsShaper):
    """
    An action shaper is used to convert an agent's model output to an Action object which can be executed by the
    environment.
    """
    @abstractmethod
    def __call__(self, model_action, decision_event, snapshot_list):
        pass

    def reset(self):
        pass
