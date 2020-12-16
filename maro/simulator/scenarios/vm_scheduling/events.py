# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Events(Enum):
    """VM-PM pairs related events."""
    # VM request events.
    REQUEST = "vm_required"
    # VM finish events.
    FINISH = "vm_finished"
