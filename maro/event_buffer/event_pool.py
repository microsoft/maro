# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .atom_event import AtomEvent
from .cascade_event import CascadeEvent
from .event_linked_list import EventLinkedList
from .event_state import EventState
from .typings import Event, EventList, List, Union


class EventPool:
    """Event pool used to generate and pool event object.

    The pooling function is disabled by default, then it is used as an Event generator with a buffer.

    When enable pooling, it will recycle events.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._atom_pool: List[AtomEvent] = []
        self._cascade_pool: List[CascadeEvent] = []
        self._recycle_buffer: EventList = []

        self._event_id: int = 0

    def gen(self, tick: int, event_type: object, payload: object, is_cascade: bool = False) -> Event:
        """Generate an event.

        Args:
            tick (int): Tick of the event will be trigger.
            event_type (object): Type of new event.
            payload (object): Payload attached to this event.
            is_cascade (bool): Is the new event is cascade event.

        Returns:
            Event: AtomEvent or CascadeEvent instance.
        """
        if is_cascade:
            event = self._pop(self._cascade_pool, CascadeEvent)
        else:
            event = self._pop(self._atom_pool, AtomEvent)

        event.tick = tick
        event.event_type = event_type
        event.payload = payload
        event.id = self._event_id
        event.state = EventState.PENDING

        self._event_id += 1

        return event

    def recycle(self, events: Union[Event, EventList]):
        """Recycle specified event for further using.

        Args:
            events (Union[Event, EventList]): Event object(s) to recycle.
        """
        if type(events) != list and type(events) != EventLinkedList:
            events = [events]

        for event in events:
            if event is not None:
                self._append(event)

    def _append(self, event: Event):
        """Append event to related pool"""
        if event:
            # Deattach the payload before recycle.
            event.payload = None
            event._next_event_ = None
            event.state = EventState.FINISHED

            if isinstance(event, CascadeEvent):
                self._cascade_pool.append(event)
            else:
                self._atom_pool.append(event)

    def _pop(self, cntr: EventList, event_cls_type: type):
        """Pop an event from related pool, generate buffer events if not enough."""
        if len(cntr) == 0:
            return event_cls_type(None, None, None, None)
        else:
            return cntr.pop()
