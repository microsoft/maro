# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class ExperienceKey(Enum):
    STATE = "state"
    ACTION = "action"
    REWARD = "reward"
    NEXT_STATE = "next_state"
    NEXT_ACTION = "next_action"


class ExperienceInfoKey(Enum):
    DISCOUNT = "discount"
    TD_ERROR = "td_error"


class TransitionInfoKey(Enum):
    AGENT_ID = "agent_id"
    EVENT = "event"
    METRICS = "metrics"

