# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

from .atom_event import AtomEvent
from .cascade_event import CascadeEvent

Event = Union[AtomEvent, CascadeEvent]

EventList = List[Event]
