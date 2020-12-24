# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Events(Enum):
    """Event type for CIM problem."""
    # RELEASE_EMPTY = 10
    RETURN_FULL = "return_full"
    LOAD_FULL = "load_full"
    DISCHARGE_FULL = "discharge_full"
    # RELEASE_FULL = 14
    RETURN_EMPTY = "return_empty"
    ORDER = "order"
    VESSEL_ARRIVAL = "vessel_arrival"
    VESSEL_DEPARTURE = "vessel_departure"
    PENDING_DECISION = "pending_decision"
    LOAD_EMPTY = "load_empty"
    DISCHARGE_EMPTY = "discharge_empty"
