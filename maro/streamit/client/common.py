# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import IntEnum


class MessageType(IntEnum):
    """Message types, used to identify type of message."""
    Experiment = 0
    Episode = 1
    Tick = 2
    Data = 3
    File = 4

    Close = 10
