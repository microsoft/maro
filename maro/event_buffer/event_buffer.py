# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict
from typing import Callable, List

from .atom_event import AtomEvent
from .cascade_event import CascadeEvent
from .event_state import EventState
from .maro_events import MaroEvents
from .execute_stack import ExecuteStack


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

    def __init__(self):
        # id for events that generate from this instance
        self._id = 0
        self._pending_events = defaultdict(list)
        self._handlers = defaultdict(list)

        # used to hold all the events that been processed
        self._finished_events = []

        # index of current pending event
        self._current_index = 0

        # stack of current executng event
        self._exec_stack = ExecuteStack()

    def get_finished_events(self) -> list:
        """Get all the processed events, call this function before reset method.

        Returns:
           list: List of event object.
        """
        return self._finished_events

    def get_pending_events(self, tick: int) -> list:
        """Get pending event at specified tick.

        Args:
            Tick (int): tick of events to get.

        Returns:
            list: List of event object.
        """
        return [evt for evt in self._pending_events[tick] if evt is not None]

    def reset(self):
        """Reset internal states, this method will clear all events.

        NOTE:
            After reset the get_finished_event method will return empty list.
        """
        # remove current list memory
        for _, pending_pool in self._pending_events.items():
            pending_pool.clear()

        self._finished_events = []
        self._current_index = 0
        self._exec_stack.clear()

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
        self._id += 1

        return AtomEvent(self._id, tick, event_type, payload)

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
        self._id += 1

        return CascadeEvent(self._id, tick, event_type, payload)

    def gen_decision_event(self, tick: int, payload: object) -> CascadeEvent:
        """Generate a decision event that will stop current simulation, and ask agent for action.

        Args:
            tick (int): Tick that the event will be processed.
            payload (object): Payload of event, used to pass data to handlers.
        Returns:
            Event: Event object
        """
        return self.gen_event(tick, MaroEvents.DECISION_EVENT, payload, True)

    def gen_action_event(self, tick: int, payload: object) -> CascadeEvent:
        """Generate an event that used to dispatch action to business engine.

        Args:
            tick (int): Tick that the event will be processed.
            payload (object): Payload of event, used to pass data to handlers.
        Returns:
            Event: Event object
        """
        return self.gen_event(tick, MaroEvents.TAKE_ACTION, payload, True)

    def gen_event(self, tick: int, event_type: object, payload: object, is_cascade: bool):
        """Generate an event object.

        Args:
            tick (int): Tick that the event will be processed.
            event_type (object): Type of this event.
            payload (object): Payload of event, used to pass data to handlers.
            is_cascade (bool): If the event is a cascade event, default is True
        Returns:
            Event: Event object
        """
        if is_cascade:
            return self.gen_cascade_event(tick, event_type, payload)
        else:
            return self.gen_atom_event(tick, event_type, payload)

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

    def insert_event(self, event):
        """Insert an event to the pending queue.

        Args:
            event (Event): Event to insert,
                usually get event object from get_atom_event or get_cascade_event.
        """

        self._pending_events[event.tick].append(event)

    def execute(self, tick: int) -> list:
        """Process and dispatch event by tick.

        NOTE:
            The executing process will be stopped if there is any cascade event,
            and all following cascade events will be returned,
            so should check if the return list is empty before step to next tick.

        Args:
            tick (int): Tick used to process events.

        Returns:
            list: Pending cascade event list at current point.
        """

        if tick in self._pending_events:
            cur_events = self._pending_events[tick]

            # 1. check if current events match tick
            while self._current_index < len(cur_events) or len(self._exec_stack) != 0:
                next_events = None

                # 2. check if executing stack if empty
                if len(self._exec_stack) != 0:
                    # pop from stack
                    next_events = self._exec_stack.pop()

                if next_events is None and self._current_index < len(cur_events):
                    event = cur_events[self._current_index]
                    # 3. push into stack,
                    #    we only process cascade events that in pending state
                    is_decision_event = event.event_type == MaroEvents.DECISION_EVENT

                    pending_exec_events = []

                    for j in range(self._current_index, len(cur_events)):
                        if (cur_events[j].event_type == MaroEvents.DECISION_EVENT) == is_decision_event:
                            pending_exec_events.append(cur_events[j])
                            cur_events[self._current_index] = None

                            self._current_index += 1
                        else:
                            break

                    for _ in range(len(pending_exec_events)):
                        self._exec_stack.push(pending_exec_events.pop())

                    next_events = self._exec_stack.pop()

                if next_events is None:
                    break

                if isinstance(next_events, list):
                    return next_events

                if next_events.event_type and next_events.event_type in self._handlers:
                    handlers = self._handlers[next_events.event_type]

                    for handler in handlers:
                        handler(next_events)

                next_events.state = EventState.FINISHED

                self._finished_events.append(next_events)

            # reset
            self._current_index = 0
        return []
