# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import IntEnum


class Events(IntEnum):
    """Event type for CIM problem."""
    # RELEASE_EMPTY = 10
    RETURN_FULL = 11
    LOAD_FULL = 12
    DISCHARGE_FULL = 13
    # RELEASE_FULL = 14
    RETURN_EMPTY = 15
    ORDER = 16
    VESSEL_ARRIVAL = 17
    VESSEL_DEPARTURE = 18
    PENDING_DECISION = 19
    LOAD_EMPTY = 20
    DISCHARGE_EMPTY = 21