# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Events(Enum):
    """VM-PM pairs related events."""
    # VM requirement events.
    REQUIREMENTS = "vm_required"
    # VM finished events.
    FINISHED = "vm_finished"
