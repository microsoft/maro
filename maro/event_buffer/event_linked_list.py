# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Optional, Union

from .event import AbsEvent, ActualEvent, CascadeEvent, DummyEvent
from .event_state import EventState
from .maro_events import MaroEvents


class EventLinkedList:
    """Event linked list used to provide methods for easy accessing.

    Event linked list only support 2 methods to add event:

    1. append: Append event to the end.
    2. insert: Insert event to the head.

    Pop method used to pop event(s) from list head, according to current event type,
    it may return a list of decision event, or just an AtomEvent object.

    .. code-block:: python

        event_list = EventLinkedList()

        # Append a new event to the end
        event_list.append(my_event)

        # Insert a event to the head
        event_list.insert(my_event_2)

        # Pop first event
        event = event_list.pop()
    """

    def __init__(self) -> None:
        # Head & tail of events.
        self._head: DummyEvent = DummyEvent()
        self._tail: AbsEvent = self._head
        self._count: int = 0

        # Used to support for loop.
        self._iter_cur_event: Optional[AbsEvent] = None

    def clear(self) -> None:
        """Clear current events."""

        # We just drop the next events reference, GC or EventPool will collect them.
        self._head.next_event = None
        self._tail = self._head
        self._count = 0

    def append_tail(self, event: ActualEvent) -> None:
        """Append an event to the end.

        Args:
            event (Event): New event to append.
        """
        # Link to the tail, update the tail.
        self._tail.next_event = event
        self._tail = event
        self._count += 1

    def append(self, event: ActualEvent) -> None:
        """Alias for append_tail.

        Args:
            event (Event): New event to append.
        """
        self.append_tail(event)

    def append_head(self, event: ActualEvent) -> None:
        """Insert an event to the head, will be the first one to pop.

        Args:
            event (Event): Event to insert.
        """
        # Link to head, update head.
        if self._count == 0:
            self.append_tail(event)
        else:
            event.next_event = self._head.next_event
            self._head.next_event = event
            self._count += 1

    def _extract_sub_events(self, event: CascadeEvent) -> None:
        """Extract sub events (immediate events) of CascadeEvent to the head.
        """
        # Make immediate event list as the head of current list.
        event.immediate_event_tail.next_event = self._head.next_event
        self._head.next_event = event.immediate_event_head.next_event
        self._count += event.immediate_event_count
        event.clear()

    def _clear_finished_events(self) -> None:
        """Remove all finished events from the head of the list.
        """
        def _is_finish(event: ActualEvent) -> bool:
            return event.state in (EventState.FINISHED, EventState.RECYCLING)

        while self._head.next_event is not None and _is_finish(self._head.next_event):
            event = self._head.next_event
            self._head.next_event = event.next_event
            self._count -= 1

            if isinstance(event, CascadeEvent) and event.immediate_event_count != 0:
                self._extract_sub_events(event)

    def _collect_pending_decision_events(self) -> List[CascadeEvent]:
        event = self._head.next_event
        decision_events = []
        while event is not None and event.event_type == MaroEvents.PENDING_DECISION:
            assert isinstance(event, CascadeEvent)
            decision_events.append(event)
            event = event.next_event
        return decision_events

    def clear_finished_and_get_front(self) -> Union[None, ActualEvent, List[ActualEvent]]:
        """Clear all finished events in the head of the list
        and then get the first event that its state is not Finished.

        Returns:
            Union[Event, EventList]: A list of decision events if current event is a decision event, or an AtomEvent.
        """

        self._clear_finished_events()

        if self._head.next_event is None:
            return None
        elif any([
            self._head.next_event.state == EventState.EXECUTING,
            self._head.next_event.event_type != MaroEvents.PENDING_DECISION
        ]):
            return self._head.next_event
        else:
            return self._collect_pending_decision_events()

    def __len__(self):
        """Length of current list."""
        return self._count

    def __iter__(self):
        """Beginning of 'for' loop."""
        self._iter_cur_event = self._head
        return self

    def __next__(self):
        """Get next item for 'for' loop."""
        if self._iter_cur_event.next_event is None:
            raise StopIteration()

        self._iter_cur_event = self._iter_cur_event.next_event
        return self._iter_cur_event
