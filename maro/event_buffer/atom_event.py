# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .event_state import EventState


class AtomEvent:
    """Basic event object that used to hold information that for callback.

    Note:
        The payload of event can be any object that related with specified logic.

    Args:
        id (int): Id of this event.
        tick (int): Tick that this event will be processed.
        event_type (int): Type of this event, this is a customize field,
            there is one predefined event type is 0 (PREDEFINE_EVENT_ACTION).
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

    def __init__(self, id: int, tick: int, event_type: object, payload: object):
        self.id = id
        self.tick = tick
        self.payload = payload
        self.event_type = event_type
        self.state = EventState.PENDING

        # Used to link to next event in linked list
        self._next_event_ = None
