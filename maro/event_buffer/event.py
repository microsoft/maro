# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import abc
from typing import Optional

from .event_state import EventState


class AbsEvent(metaclass=abc.ABCMeta):
    """Abstract interface for events. Hold information that for callback.

    Note:
        The payload of event can be any object that related with specified logic.

    Args:
        id (int): Id of this event.
        tick (int): Tick that this event will be processed.
        event_type (int): Type of this event, this is a customized field,
            there is one predefined event type 0 (PREDEFINE_EVENT_ACTION).
        payload (object): Payload of this event.

    Attributes:
        id (int): Id of this event, usually this is used for "joint decision" node
                that need "sequential action".
        tick (int): Process tick of this event.
        payload (object): Payload of this event, can be any object.
        event_type (object): Type of this event, can be any type,
                EventBuffer will use this to match handlers.
        state (EventState): Internal life-circle state of event.
    """

    def __init__(self, id: Optional[int], tick: Optional[int], event_type: object, payload: object) -> None:
        self.id: Optional[int] = id
        self.tick: Optional[int] = tick
        self.payload: object = payload
        self.event_type: object = event_type
        self.state: EventState = EventState.PENDING

        # Used to link to next event in linked list
        self.next_event: Optional[ActualEvent] = None

    def reset_value(
        self, id: Optional[int], tick: Optional[int], event_type: object, payload: object, state: EventState
    ) -> None:
        self.id: Optional[int] = id
        self.tick: Optional[int] = tick
        self.event_type: object = event_type
        self.payload: object = payload
        self.state: EventState = state


class DummyEvent(AbsEvent):
    def __init__(self) -> None:
        # Add parameters could be set to None since the event is dummy.
        super().__init__(None, None, None, None)


class ActualEvent(AbsEvent, metaclass=abc.ABCMeta):
    def __init__(self, id: Optional[int], tick: Optional[int], event_type: object, payload: object) -> None:
        super().__init__(id, tick, event_type, payload)


class AtomEvent(ActualEvent):
    """Basic atom event without any additional functions or attributes.
    """
    def __init__(self, id: Optional[int], tick: Optional[int], event_type: object, payload: object) -> None:
        super().__init__(id, tick, event_type, payload)


class CascadeEvent(ActualEvent):
    """Event that support add immediate events (or sub events), these
    events will be execute right after its parent.

    Some times there may be some events that depend on another one,
    then you can append these events with add_immediate_event method, then
    these events will be processed after the parent event.
    """

    def __init__(self, id: Optional[int], tick: Optional[int], event_type: object, payload: object) -> None:
        super().__init__(id, tick, event_type, payload)

        # Head & tail of immediate event list.
        self._immediate_event_head: DummyEvent = DummyEvent()
        self._immediate_event_tail: Optional[ActualEvent] = None

        self._immediate_event_count = 0

    @property
    def immediate_event_count(self) -> int:
        return self._immediate_event_count

    @property
    def immediate_event_head(self) -> DummyEvent:
        return self._immediate_event_head

    @property
    def immediate_event_tail(self) -> Optional[ActualEvent]:
        return self._immediate_event_tail

    def clear(self) -> None:
        self._immediate_event_head.next_event = self._immediate_event_tail = None
        self._immediate_event_count = 0

    def add_immediate_event(self, event: ActualEvent, is_head: bool = False) -> bool:
        """Add an immediate event, that will be processed right after the current event.

        Immediate events are only supported to be inserted into the head or tail of the immediate event list.
        By default, the events will be appended to the end.

        NOTE:
            The tick of the event to insert must be the same as the current event, or will fail to insert.

        Args:
            event (ActualEvent): Event object to insert. It has to be an actual event. A dummy event is unacceptable.
            is_head (bool): Whether to insert at the head or append to the end.

        Returns:
            bool: True if success, or False.
        """
        # Make sure the immediate event's tick is the same as the current one.
        if event.tick != self.tick:
            return False

        if self._immediate_event_count == 0:
            # 'self._immediate_event_count == 0' means the linked list is empty.
            # In this case, inserting at the head is identical with appending to the end.
            self._immediate_event_head.next_event = self._immediate_event_tail = event
        elif is_head:
            assert event.next_event is None, 'Follow-up events are unacceptable when inserting the event into the head'
            event.next_event = self._immediate_event_head.next_event
            self._immediate_event_head.next_event = event
        else:
            self._immediate_event_tail.next_event = event
            self._immediate_event_tail = event

        self._immediate_event_count += 1

        return True
