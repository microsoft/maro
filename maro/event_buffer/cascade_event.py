# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .atom_event import AtomEvent


class CascadeEvent(AtomEvent):
    """Special event type that support add immediate events (or sub events), these
    events will be execute right after its parent.

    Some times there may be some events that depend on another one,
    then you can append these events into immediate_event_list, then
    these events will be processed after the main event at same tick.
    """

    def __init__(self, id: int, tick: int, event_type: object, payload: object):
        super().__init__(id, tick, event_type, payload)

        self._immediate_event_list = []

    def add_immediate_event(self, event, is_head: bool = False) -> bool:
        """Add a immediate event, that will be processed right after current event.

        Immediate event only support add to the head or tail, default will append to the end.

        NOTE:
            Tick of immediate event must same as current event, or will fail to insert.

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
            self._immediate_event_list.insert(0, event)
        else:
            self._immediate_event_list.append(event)

        return True
