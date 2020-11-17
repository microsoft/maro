# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, List

from .event_state import EventState
from .atom_event import AtomEvent
from .cascade_event import CascadeEvent


Event = Union[AtomEvent, CascadeEvent]


class EventPool:
    """Event pool used to generate and pool event object.

    The pooling function is disabled by default, then it is used as an Event generator with a buffer.

    When enable pooling, it will recycle events.
    """

    def __init__(self, enable: bool = False, capacity: int = 1000):
        self._capacity = capacity
        self._enabled: bool = enable
        self._atom_pool: List[AtomEvent] = []
        self._cascade_pool: List[CascadeEvent] = []
        self._recycle_buffer: List[Event] = []

        self._event_id: int = 0

        if enable:
            for _ in range(capacity):
                self._atom_pool.append(AtomEvent(None, None, None, None))
                self._cascade_pool.append(CascadeEvent(None, None, None, None))

    @property
    def enabled(self) -> bool:
        """bool: Is pooling enabled."""
        return self._enabled

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

    def recycle(self, evt: Event, with_buffer: bool = True):
        """Recycle specified event for further using.

        Args:
            evt (Event): Event object to recycle.
            with_buffer (bool): Is recycle object should put into buffer first?
        """
        if evt is not None:
            if with_buffer:
                # append to the end of buffer
                self._recycle_buffer.append(evt)
            else:
                self._append(evt)

    def flush(self):
        """Flush current recycle buffer, make cached events ready to use."""
        for _ in range(len(self._recycle_buffer)):
            evt = self._recycle_buffer.pop()

            self._append(evt)

    def _append(self, evt: Event):
        """Append event to related pool"""
        if evt:
            # deattach the payload before recycle
            evt.payload = None
            evt.state = EventState.FINISHED

            if isinstance(evt, CascadeEvent):
                self._cascade_pool.append(evt)
            else:
                self._atom_pool.append(evt)

    def _pop(self, cntr: List[Event], evt_type: type):
        """Pop an event from related pool, generate buffer events if not enough."""
        if len(cntr) == 0:
            for _ in range(self._capacity):
                cntr.append(evt_type(None, None, None, None))

        return cntr.pop()
