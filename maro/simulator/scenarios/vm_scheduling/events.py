# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Events(Enum):
    """VM-PM pairs related events."""
    # VM requirement events.
    REQUEST = "vm_required"
    # VM finished events.
    FINISH = "vm_finished"
