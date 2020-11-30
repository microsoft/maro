# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict
from typing import Callable

from .atom_event import AtomEvent
from .cascade_event import CascadeEvent
from .event_list import EventLinkedList
from .event_pool import EventPool
from .event_state import EventState
from .maro_events import MaroEvents
from .typings import Event, EventList


class EventBuffer:
    """
    EventBuffer used to hold events, and dispatch them at specified tick.

    NOTE:
        Different with normal event dispatcher,
        EventBuffer will stop executing and return following cascade events when it meet
        first pending cascade event, this may cause each tick is separated into several parts,
        users should check the return result before step to next tick.

        And insert order will affect the processing order,
        so ensure the order when you need something strange.
    """

    def __init__(self, with_pooling: bool = False):
        # id for events that generate from this instance
        self._pending_events = defaultdict(EventLinkedList)
        self._handlers = defaultdict(list)

        # used to hold all the events that been processed
        self._finished_events = []

        # index of current pending event
        self._current_index = 0

        self._event_pool = EventPool(with_pooling)

    def get_finished_events(self) -> EventList:
        """Get all the processed events, call this function before reset method.

        Returns:
            EventList: List of event object.
        """
        return self._finished_events

    def get_pending_events(self, tick: int) -> EventList:
        """Get pending event at specified tick.

        Args:
            Tick (int): tick of events to get.

        Returns:
            EventList: List of event object.
        """
        return [evt for evt in self._pending_events[tick] if evt is not None]

    def reset(self):
        """Reset internal states, this method will clear all events.

        NOTE:
            After reset the get_finished_event method will return empty list.
        """
        # collect the events from pendding and finished pool to reuse them
        self._event_pool.recycle(self._finished_events)

        self._finished_events.clear()

        for _, pending_pool in self._pending_events.items():
            self._event_pool.recycle(pending_pool)

            pending_pool.clear()

        self._current_index = 0

    def gen_atom_event(self, tick: int, event_type: object, payload: object = None) -> AtomEvent:
        """Generate an atom event, an atom event is for normal usages,
        they will not stop current event dispatching process.

        Args:
            tick (int): Tick that the event will be processed.
            event_type (object): Type of this event.
            payload (object): Payload of event, used to pass data to handlers.

        Returns:
            AtomEvent: Atom event object
        """
        return self._event_pool.gen(tick, event_type, payload, False)

    def gen_cascade_event(self, tick: int, event_type: object, payload: object) -> CascadeEvent:
        """Generate an cascade event that used to hold immediate events that
        run right after current event.

        Args:
            tick (int): Tick that the event will be processed.
            event_type (object): Type of this event.
            payload (object): Payload of event, used to pass data to handlers.

        Returns:
            CascadeEvent: Cascade event object.
        """
        return self._event_pool.gen(tick, event_type, payload, True)

    def gen_decision_event(self, tick: int, payload: object) -> CascadeEvent:
        """Generate a decision event that will stop current simulation, and ask agent for action.

        Args:
            tick (int): Tick that the event will be processed.
            payload (object): Payload of event, used to pass data to handlers.
        Returns:
            CascadeEvent: Event object
        """
        return self._event_pool.gen(tick, MaroEvents.DECISION_EVENT, payload, True)

    def gen_action_event(self, tick: int, payload: object) -> CascadeEvent:
        """Generate an event that used to dispatch action to business engine.

        Args:
            tick (int): Tick that the event will be processed.
            payload (object): Payload of event, used to pass data to handlers.
        Returns:
            CascadeEvent: Event object
        """
        return self._event_pool.gen(tick, MaroEvents.TAKE_ACTION, payload, True)

    def gen_event(self, tick: int, event_type: object, payload: object, is_cascade: bool) -> Event:
        """Generate an event object.

        Args:
            tick (int): Tick that the event will be processed.
            event_type (object): Type of this event.
            payload (object): Payload of event, used to pass data to handlers.
            is_cascade (bool): If the event is a cascade event, default is True
        Returns:
            Event: Event object
        """
        return self._event_pool.gen(tick, event_type, payload, is_cascade)

    def register_event_handler(self, event_type: int, handler: Callable):
        """Register an event with handler, when there is an event need to be processed,
        EventBuffer will invoke the handler if there are any event's type match specified at each tick.

        NOTE:
            Callback function should only hold one parameter that is event object.

        Args:
            event_type (int): Type of event that the handler want to process.
            handler (Callable): Handler that will process the event.
        """
        self._handlers[event_type].append(handler)

    def insert_event(self, event: Event):
        """Insert an event to the pending queue.

        Args:
            event (Event): Event to insert,
                usually get event object from get_atom_event or get_cascade_event.
        """

        self._pending_events[event.tick].append(event)

    def execute(self, tick: int) -> EventList:
        """Process and dispatch event by tick.

        NOTE:
            The executing process will be stopped if there is any cascade event,
            and all following cascade events will be returned,
            so should check if the return list is empty before step to next tick.

        Args:
            tick (int): Tick used to process events.

        Returns:
            EventList: Pending cascade event list at current point.
        """
        if tick in self._pending_events:
            cur_events_list: EventLinkedList = self._pending_events[tick]

            # 1. check if current events match tick
            while len(cur_events_list):
                # flush event pool
                next_events = cur_events_list.pop()

                if next_events is None:
                    # end of current tick
                    break

                # only decision event is a list (even only one item)
                # NOTE: decision event do not have handlers, and simulator will set its state
                # to finished after recieved an action.
                if type(next_events) == list:
                    return next_events
                else:
                    next_events.state = EventState.EXECUTING

                # 3. invoke handlers
                if next_events.event_type and next_events.event_type in self._handlers:
                    handlers = self._handlers[next_events.event_type]

                    for handler in handlers:
                        handler(next_events)

                # finish events, so execute stack will extract its immediate events
                next_events.state = EventState.FINISHED

                if self._event_pool.enabled:
                    self._event_pool.recycle(next_events)
                else:
                    self._finished_events.append(next_events)
        return []
