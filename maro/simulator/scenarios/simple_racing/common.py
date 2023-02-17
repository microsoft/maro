# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.common import BaseAction, BaseDecisionEvent


class Action(BaseAction):
    # TODO: You can update the Action class, and process it in SimpleRacing._on_action_received() correspondingly.
    def __init__(self, action: np.ndarray) -> None:
        """
        Args:
            action (np.ndarray): The action defined in Gym Env. Or said the one deliver to EymEnv.step() function.
        """
        self.action = action


class DecisionEvent(BaseDecisionEvent):
    # TODO: You can update the DecisionEvent class, and create it in SimpleRacing.step() correspondingly.
    def __init__(self, state: np.ndarray) -> None:
        """
        Args:
            state (np.ndarray): The observation defined in Gym Env. Or said the return value of EymEnv.step() function.
        """
        self.state = state
