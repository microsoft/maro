# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.common import BaseAction, BaseDecisionEvent


class Action(BaseAction):
    def __init__(self, action) -> None:
        self.action = action


class DecisionEvent(BaseDecisionEvent):
    def __init__(self, state) -> None:
        self.state = state
