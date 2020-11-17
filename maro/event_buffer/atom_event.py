# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .event_state import EventState


class AtomEvent:
    """Basic event object that used to hold information that for callback.

    Note:
        The payload of event can be any object that related with specified logic.

    Args:
        tick (int): Tick that this event will be processed.
        event_type (int): Type of this event, this is a customize field,
            there is one predefined event type is 0 (PREDEFINE_EVENT_ACTION).
        payload (object): Payload of this event.
        category (EventCategory): Category mark of this event.

    Attributes:
        id (int): Id of this event, usually this is used for "joint decision" node
                that need "sequential action".
        tick (int): Process tick of this event.
        payload (object): Payload of this event, can be any object.
        category (EventCategory): Category of this event.
        event_type (object): Type of this event, can be any type, usually int,
                EventBuffer will use this to match handlers.
        state (EventState): Internal life-circle state of event.
        immediate_event_list (List[Event]): Used to hold events that depend on current event,
                these events will be processed after this event.
    """

    def __init__(self, id: int, tick: int, event_type: object, payload: object):
        self.id = id
        self.tick = tick
        self.payload = payload
        self.event_type = event_type
        self.state = EventState.PENDING

    def __repr__(self):
        return f"{{ tick: {self.tick}, type: {self.event_type}, " \
               f"state: {self.state}, payload: {self.payload} }}"

    def __str__(self):
        return self.__repr__()
