# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, List
from .event_state import EventState
from .atom_event import AtomEvent
from .cascade_event import CascadeEvent
from .maro_events import MaroEvents


Event = Union[AtomEvent, CascadeEvent]


class ExecuteStack:
    def __init__(self):
        self._cntr: List[Event] = []

    def push(self, evt: Event):
        # we do not extract current event when pushing,
        # just put to the top
        self._cntr.append(evt)

    def pop(self) -> Union[Event, List[Event]]:
        while(len(self._cntr) > 0):
            # NOTE: we do not pop when popping, only when the state changed to finished
            # NOTE: we do not maintain the event state here
            evt: Event = self._cntr[-1]

            # for finished event, will extract its sub-events first, then ignore it
            if evt.state == EventState.FINISHED:
                self._cntr.pop()
                self._extract_sub_events(evt)

                # ignore this finished event
                continue

            # 2. check if there is any decision events
            if evt.event_type != MaroEvents.DECISION_EVENT:
                # 2.1 normal event, just pop this one
                return evt
            else:
                # 2.2 return decision events in order
                decision_events = []

                next_evt: Event = None
                index: int = len(self._cntr) - 1

                # find all following pending decision event
                while(evt is not None and evt.state == EventState.PENDING and evt.event_type == MaroEvents.DECISION_EVENT):
                    decision_events.append(evt)

                    index -= 1

                    evt = self._cntr[index] if index >= 0 else None

                return decision_events

        return None

    def clear(self):
        self._cntr.clear()

    def __len__(self):
        return len(self._cntr)

    def _extract_sub_events(self, evt: Event):
        if isinstance(evt, CascadeEvent) and evt._immediate_event_list is not None:
            # push from back
            for i in range(len(evt._immediate_event_list)):
                self._cntr.append(evt._immediate_event_list.pop())
