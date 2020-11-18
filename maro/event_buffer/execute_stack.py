# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
from math import floor
from typing import List, Union

from .atom_event import AtomEvent
from .cascade_event import CascadeEvent
from .event_state import EventState
from .maro_events import MaroEvents

Event = Union[AtomEvent, CascadeEvent]


class ExecuteStack:
    """Stack used to hold events pending to execute, and extract immediate events with a better performance."""

    def __init__(self):
        self._cntr: List[Event] = []

        # Used to hold start index of reversing
        self._reverse_start_index = -1

    def push(self, evt: Event):
        """Push an event on the top.

        Args:
            evt (Event): Event to push, must be a AtomEvent or Cascade event.
        """
        # we do not extract current event when pushing,
        # just put to the top
        self._cntr.append(evt)

    def pop(self) -> Union[Event, List[Event]]:
        """Pop not finished event(s).

        NOTE:
            1. This method will return a list of following DecisionEvent if the event on top
                is DecisionEvent.
            2. Different with normal stack pop method, this function will not remove the
            reference of the poped events until theirs state is finished.

        Returns:
            Union[Event, List[Event]]: Normal event, or a list of following DecisionEvent.
        """
        while(len(self._cntr) > 0):
            # NOTE: we do not pop when popping, only when the state changed to finished
            # NOTE: we do not maintain the event state here
            evt: Event = self._cntr[-1]

            # for finished event, will extract its sub-events first, then ignore it
            if evt.state == EventState.FINISHED or evt.state == EventState.RECYCLING:
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

                index: int = len(self._cntr) - 1

                # find all following pending decision event
                while(evt is not None and evt.state == EventState.PENDING and evt.event_type == MaroEvents.DECISION_EVENT):
                    decision_events.append(evt)

                    index -= 1

                    evt = self._cntr[index] if index >= 0 else None

                return decision_events

        return None

    def clear(self):
        """Clear execute stack."""
        self._reverse_start_index = -1
        self._cntr.clear()

    @contextlib.contextmanager
    def reverse_push(self):
        """Used to reverse following events after within current context.

        Returns:
            AbstractContextManager: Context manager for 'with' statement.
        """
        # keep current size as reverse start
        self._reverse_start_index = len(self._cntr)

        yield self

        # do reverse
        cntr_length = len(self._cntr)
        stop = floor((cntr_length - self._reverse_start_index) / 2)

        for i in range(stop):
            start = self._reverse_start_index + i
            end = cntr_length - i - 1

            self._cntr[start], self._cntr[end] = self._cntr[end], self._cntr[start]

    def __len__(self) -> int:
        """Length of current stack.

        Returns:
            int: Length of current stack.
        """
        return len(self._cntr)

    def _extract_sub_events(self, evt: Event):
        """Extract specified event's immediate event list, and push to current stack."""
        if isinstance(evt, CascadeEvent) and evt._immediate_event_list is not None:
            # push from back
            for i in range(len(evt._immediate_event_list)):
                self._cntr.append(evt._immediate_event_list.pop())
