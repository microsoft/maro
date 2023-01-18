# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.common import BaseAction, BaseDecisionEvent


class Action(BaseAction):
    def __init__(self, action: np.ndarray) -> None:
        self.action = action


class DecisionEvent(BaseDecisionEvent):
    def __init__(self, state: np.ndarray) -> None:
        self.state = state
