# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum, IntEnum


class Events(Enum):
    """VM-PM pairs related events."""
    # VM request events.
    REQUEST = "vm_required"


class PostponeType(Enum):
    """Postpone type."""
    # Postpone the VM requirement due to the resource exhaustion.
    Resource = "resource"
    # Postpone the VM requirement due to the agent's decision.
    Agent = "agent"


class PmState(IntEnum):
    """PM oversubscription state, includes empty, oversubscribable, non-oversubscribable."""
    NON_OVERSUBSCRIBABLE = -1
    EMPTY = 0
    OVERSUBSCRIBABLE = 1


class VmCategory(IntEnum):
    DELAY_INSENSITIVE = 0
    INTERACTIVE = 1
    UNKNOWN = 2
