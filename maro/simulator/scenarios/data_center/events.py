# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class DataCenterEvents(Enum):
    """VM-PM pairs related events."""
    # VM requirement events
    REQUIRE = "vm_required"
    # VM finished events
    FINISHED = "vm_finished"
