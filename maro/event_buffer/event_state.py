# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import IntEnum


class EventState(IntEnum):
    """State of event for internal using.
    """
    PENDING = 0
    EXECUTING = 1
    FINISHED = 2
    # Will be recycled
    RECYCLING = 3
