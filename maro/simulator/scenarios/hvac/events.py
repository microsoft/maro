# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Events(Enum):
    AHU_SET = "AHU_set"
    PENDING_DECISION = "pending_decision"
