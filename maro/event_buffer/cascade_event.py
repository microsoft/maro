# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .atom_event import AtomEvent


class CascadeEvent(AtomEvent):
    """Special event type that support add immediate events (or sub events), these
    events will be execute right after its parent.

    Some times there may be some events that depend on another one,
    then you can append these events into immediate_event_list, then
    these events will be processed after the parent event at same tick.
    """

    def __init__(self, id: int, tick: int, event_type: object, payload: object):
        super().__init__(id, tick, event_type, payload)

        # Header of immediate event list
        self._immediate_event_head = AtomEvent(None, None, None, None)

        # Pointer to last immediate event, used for speed up immediate event extract
        self._last_immediate_event = self._immediate_event_head
        self._immediate_event_count = 0

    def add_immediate_event(self, event, is_head: bool = False) -> bool:
        """Add an immediate event, that will be processed right after the current event.

        Immediate events are only supported to be inserted into the head or tail of the immediate event list.
        By default, the events will be appended to the end.

        NOTE:
            The tick of the event to insert must be the same as the current event, or will fail to insert.

        Args:
            event (Event): Event object to insert.
            is_head (bool): If insert into the head (0 index), or append to the end.

        Returns:
            bool: True if success, or False.
        """
        # Make sure immediate event's tick same as current
        if event.tick != self.tick:
            return False

        if is_head:
            event._next_event_ = self._immediate_event_head._next_event_

            self._immediate_event_head._next_event_ = event
        else:
            self._last_immediate_event._next_event_ = event

            self._last_immediate_event = event

        self._immediate_event_count += 1

        return True
