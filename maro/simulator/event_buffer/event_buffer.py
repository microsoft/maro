# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from heapq import heappush, heappop
from enum import IntEnum
from collections import defaultdict
from typing import List, Dict, Callable

# this is our predefine event type for action processing
DECISION_EVENT = 0
"""int: Predefined decision event type, if business engine need to process actions from agent, then it must register this event"""


class EventTag(IntEnum):
    """
    Event tag that mark the usage of event
    """
    ATOM = 0
    CASCADE = 1


class EventState(IntEnum):
    """
    State of event
    """
    PENDING = 0
    EXECUTING = 1
    FINISHED = 2


class Event:
    """Event object that used to hold information that for callback.

    Note:
        The payload of event can be any object that related with specified logic

    Args:
        tick (int): tick that this event will be processed
        event_type (int): type of this event, this is a customize field, there is one predefined event type is 0 (PREDEFINE_EVENT_ACTION)
        payload (object): payload of this event
        tag (EventTag): tag mark of this event
    """
    def __init__(self, tick: int, event_type: int, payload, tag: int, priority: int = 0):
        self.tick = tick
        self.priority = priority
        self.payload = None
        self.source = None
        self.target = None
        self.payload = payload
        self.immediate_event_list = []
        self.tag = tag
        self.event_type = event_type
        self.state = EventState.PENDING

    def __lt__(self, other):
        # since heapq is small endian heap, so we expect self large than other, that we can make bigger number has higher priority
        return self.priority > other.priority

    def __repr__(self):
        return f"{{ tick: {self.tick}, type: {self.event_type}, " \
               f"tag: {self.tag}, state: {self.state}, payload: {self.payload} }}"


class EventBuffer:
    """
    EventBuffer used to hold the events, and dispatch them at specified tick.
    """
    def __init__(self):
        self._pending_events = defaultdict(list)
        self._pending_buffer = [] # used to hold poped cascade events
        self._handlers = defaultdict(list)
        self._finished_events = []  # used to hold all the events that being processed

    def get_finished_events(self) -> List[Event]:
        """Get all the processed events, call this function before reset method

        Returns:
            List[Event]: list of Event object
        """
        return self._finished_events

    def get_pending_events(self, tick: int) -> List[Event]:
        """Get pending event at specified tick.
        
        Args:
            tick (int): tick of events to get

        Returns:
            List[Event]: list of Event object
        """
        return self._pending_events[tick]

    def restore(self):
        """Restore states

        Note:
            Not implemented
        """
        # next version
        pass

    def dump(self):
        """Dump events

        Note:
            Not implemented.
        """
        # next version
        pass

    def reset(self):
        """Reset the state to initial

        NOTE: 
            After reset the get_finished_event method will return empty list
        """
        self._pending_events = defaultdict(list)
        self._pending_buffer = []
        self._finished_events = []

    def gen_atom_event(self, tick: int, event_type: int, payload: object, priority: int = 0) -> Event:
        """Generate an atom event

        Args:
            tick (int): tick that the event will be processed
            event_type (int): type of this event
            payload (object): payload of event, used to pass data to handlers
            priority (int): priority of this event, larger number has higher priority (executed first in same tick)

        Returns:
            Event: Event object with ATOM tag
        """
        return Event(tick, event_type, payload, EventTag.ATOM, priority)

    def gen_cascade_event(self, tick: int, event_type: int, payload: object, priority: int = 0) -> Event:
        """Generate an cascade event that used to retrieve action from agent

        Args:
            tick (int): tick that the event will be processed
            event_type (int): type of this event
            payload (object): payload of event, used to pass data to handlers
            priority (int): priority of this event, larger number has higher priority (executed first in same tick)

        Returns:
            Event: Event object with CASCADE tag
        """
        return Event(tick, event_type, payload, EventTag.CASCADE, priority)

    def register_event_handler(self, event_type: int, handler: Callable):
        """Register an event with handler, when there is an event need to be processed, EventBuffer will invoke the handler

        Args:
            event_type (int): type of event that the handler want to process
            handler (Callable): handler that will process the event
        """
        self._handlers[event_type].append(handler)
    
    def insert_event(self, event):
        """Insert an event to the pending queue

        Args:
            event (Event): event to insert, usually get event object from get_atom_event or get_cascade_event
        """
        #self._pending_events[event.tick].append(event)
        target_queue = self._pending_events[event.tick]

        # we use lenght of current queue to make sure the insert order with same prority will not be changed
        heappush(target_queue, (event, len(target_queue)))

    def execute(self, tick) -> List[Event]:
        """Process and dispatch event by tick

        Args:
            tick (int): tick used to process events

        Returns:
            List[Event]: pending cascade event list
        """
        # if there is event in pending pool, we should due with it
        if len(self._pending_buffer) > 0:
            self._process_event(self._pending_buffer[0])

            del self._pending_buffer[0]

            if len(self._pending_buffer) > 0 and self._pending_buffer[0].state == EventState.PENDING:
                # if there is any pending cascade event, then process it 
                return self._pending_buffer
            else:
                # or finish them, and add into finish pool
                for event in self._pending_buffer:
                    self._finished_events.append(event)

        # process by ticks
        if tick in self._pending_events:
            cur_events = self._pending_events[tick]

            # 1. check if current events match tick
            while len(cur_events) > 0:
                event, _ = cur_events[0]

                # 2. check if it is a cascade event and its state, we only process cascade events that in pending state
                if event.tag == EventTag.CASCADE and event.state == EventState.PENDING:
                    # NOTE: here we return all the cascade events next to current one

                    while len(cur_events) > 0:
                        evt, _ = cur_events[0]

                        if evt.tag == EventTag.CASCADE:
                            self._pending_buffer.append(evt)

                            # pop as we find the result
                            heappop(cur_events)
                        else:
                            break

                    return self._pending_buffer
                else:
                    # or pop and process it
                    heappop(cur_events)
                
                    self._process_event(event)

        return []

    def _process_event(self, event: Event):
        # 3. or it is an atom event, just invoke the handlers
        if event.state != EventState.FINISHED:
            # 3.1. if handler exist
            if event.event_type and event.event_type in self._handlers:
                handlers = self._handlers[event.event_type]

                for handler in handlers:
                    handler(event)

            # 3.2. sub events
            for sub_event in event.immediate_event_list:
                if sub_event.event_type in self._handlers:
                    handlers = self._handlers[sub_event.event_type]

                    for handler in handlers:
                        handler(sub_event)

                    sub_event.state = EventState.FINISHED

            event.state = EventState.FINISHED

        self._finished_events.append(event)

                    


