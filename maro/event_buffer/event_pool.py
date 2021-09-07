# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from itertools import count
from typing import Iterable, Iterator, List, Union

from .event import ActualEvent, AtomEvent, CascadeEvent
from .event_linked_list import EventLinkedList
from .event_state import EventState


def _pop(cntr: List[ActualEvent], event_cls_type: type) -> ActualEvent:
    """Pop an event from related pool, generate buffer events if not enough."""
    return event_cls_type(None, None, None, None) if len(cntr) == 0 else cntr.pop()


class EventPool:
    """Event pool used to generate and pool event object.

    The pooling function is disabled by default, then it is used as an Event generator with a buffer.

    When enable pooling, it will recycle events.
    """

    def __init__(self):
        self._atom_events: List[AtomEvent] = []
        self._cascade_events: List[CascadeEvent] = []

        self._event_count: Iterator[int] = count()

    @property
    def atom_event_count(self) -> int:
        return len(self._atom_events)

    @property
    def cascade_event_count(self) -> int:
        return len(self._cascade_events)

    def gen(
        self, tick: int, event_type: object, payload: object,
        is_cascade: bool = False
    ) -> ActualEvent:
        """Generate an event.

        Args:
            tick (int): Tick of the event will be trigger.
            event_type (object): Type of new event.
            payload (object): Payload attached to this event.
            is_cascade (bool): Is the new event is cascade event.

        Returns:
            Event: AtomEvent or CascadeEvent instance.
        """
        event = _pop(self._cascade_events, CascadeEvent) if is_cascade else _pop(self._atom_events, AtomEvent)
        event.reset_value(
            id=next(self._event_count), tick=tick, event_type=event_type,
            payload=payload, state=EventState.PENDING
        )
        return event

    def recycle(self, events: Union[ActualEvent, List[ActualEvent], EventLinkedList]) -> None:
        """Recycle specified event for further using.

        Args:
            events (Union[Event, EventList]): Event object(s) to recycle.
        """
        self._append(events) if isinstance(events, ActualEvent) else self._extend(events)

    def _extend(self, events: Iterable[ActualEvent]) -> None:
        for event in events:
            self._append(event)

    def _append(self, event: ActualEvent) -> None:
        """Append event to related pool"""
        if isinstance(event, ActualEvent):
            # Detach the payload before recycle.
            event.payload = None
            event.next_event = None
            event.state = EventState.RECYCLING

            if isinstance(event, CascadeEvent):
                self._cascade_events.append(event)
            else:
                assert isinstance(event, AtomEvent)
                self._atom_events.append(event)
