# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class MaroEvents (Enum):
    """Predefined decision event types, that used to communicate with outside."""
    PENDING_DECISION = "maro_event_pending_decision"
    TAKE_ACTION = "maro_event_take_action"
