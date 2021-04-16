# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

class Experience(Enum):
    STATE = "STATE"
    ACTION = "ACTION"
    REWARD = "REWARD"
    NEXT_STATE = "NEXT_STATE"
