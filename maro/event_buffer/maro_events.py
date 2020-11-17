# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class MaroEvents (Enum):
    """Predefined decision event types, that used to communicate with outside."""
    DECISION_EVENT = "maro_event_decision_event"
    TAKE_ACTION = "maro_event_take_action"
