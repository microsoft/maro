# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .atom_event import AtomEvent
from .cascade_event import CascadeEvent
from .event_state import EventState
from .maro_events import MaroEvents
from .typings import Event, EventList, Union


class EventLinkedList:
    """Event linked list used to provide methods for easy accessing.

    Event linked list only support 2 methods to add event:

    1. append: Append event to the end.
    2. insert: Insert event to the head.

    Pop method used to pop event(s) from list head, according to current event type,
    it may return a list of deicision event, or just an AtomEvent object.

    .. code-block:: python

        event_list = EventLinkedList()

        # Append a new event to the end
        event_list.append(my_event)

        # Insert a event to the head
        event_list.insert(my_event_2)

        # Pop first event
        event = event_list.pop()
    """
    def __init__(self):
        # Head of events.
        self._head = AtomEvent(None, None, None, None)

        # Tail of events.
        self._tail = self._head

        # Current events count.
        self._count = 0

        # Used to support for loop.
        self._iter_cur_event: Event = None

    def clear(self):
        """Clear current events."""

        # We just drop the next events reference, GC or EventPool will collect them.
        self._head._next_event_ = None
        self._tail = self._head
        self._count = 0

    def append(self, event: Event):
        """Append an event to the end.

        Args:
            event (Event): New event to append.
        """
        # Link to the tail, update the tail.
        self._tail._next_event_ = event
        self._tail = event

        # Counting.
        self._count += 1

    def insert(self, event: Event):
        """Insert an event to the head, will be the first one to pop.

        Args:
            event (Event): Event to insert.
        """
        # Link to head, update head.
        event._next_event_ = self._head._next_event_
        self._head._next_event_ = event

        # Counting.
        self._count += 1

    def pop(self) -> Union[Event, EventList]:
        """Pop first event that its state is not Finished.

        Returns:
            Union[Event, EventList]: A list of decision events if current event is decision event, or an AtomEvent.
        """
        event: Event = self._head._next_event_

        while event is not None:
            # We will remove event from list until its state is FINISHED.
            if event.state == EventState.FINISHED:
                # Remove it (head).
                self._head._next_event_ = event._next_event_

                event._next_event_ = None

                # Counting.
                self._count -= 1

                # Extract sub events, this will change the head.
                self._extract_sub_events(event)

                event = self._head._next_event_

                continue

            if event.state == EventState.EXECUTING:
                return event

            if event.event_type == MaroEvents.PENDING_DECISION:
                decision_events = [event]

                # Find following decision events.
                next_event: Event = event._next_event_

                while next_event is not None and next_event.event_type == MaroEvents.PENDING_DECISION:
                    decision_events.append(next_event)

                    next_event = next_event._next_event_

                return decision_events
            else:
                return event

        return None

    def __len__(self):
        """Length of current list."""
        return self._count

    def __iter__(self):
        """Beginning of for loopping."""
        self._iter_cur_event = self._head

        return self

    def __next__(self):
        """Get next item for 'for' loopping."""
        event: Event = None

        if self._iter_cur_event is None:
            raise StopIteration()

        if self._iter_cur_event is not None:
            event = self._iter_cur_event._next_event_

            if event is None:
                raise StopIteration()
            else:
                self._iter_cur_event = event._next_event_

        return event

    def _extract_sub_events(self, event: Event):
        """Extract sub events (immediate events) of CascadeEvent to the head.

        Args:
            event (Event): Event to extract.
        """
        if type(event) == CascadeEvent:
            # Make immediate event list as the head of current list.
            if event._last_immediate_event is not None:
                event._last_immediate_event._next_event_ = self._head._next_event_
                self._head._next_event_ = event._immediate_event_head._next_event_

                self._count += event._immediate_event_count

                # Clear the reference for finished event.
                event._immediate_event_head._next_event_ = None
                event._last_immediate_event = None
