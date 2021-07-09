# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class BikeTransferPayload:
    """Payload for bike transfer event.

    Args:
        from_station_idx (int): Which station (index) this bike come from.
        to_station_idx (int): Which station (index) this bike to.
        number (int): How many bikes for current trip requirement.
    """

    summary_key = ["from_station_idx", "to_station_idx", "number"]

    def __init__(self, from_station_idx: int, to_station_idx: int, number: int = 1):
        self.from_station_idx = from_station_idx
        self.to_station_idx = to_station_idx
        self.number = number

    def __repr__(self):
        return "%s {from_station_idx: %r, to_station_idx: %r, number:%r}" % \
            (self.__class__.__name__, self.from_station_idx, self.to_station_idx, self.number)


class BikeReturnPayload:
    """Payload for bike return event.

    Args:
        from_station_idx (int): Which station (index) this bike come from.
        to_station_idx (int): Which station (index) this bike to.
        number (int): How many bikes for current trip requirement.
    """

    summary_key = ["from_station_idx", "to_station_idx", "number"]

    def __init__(self, from_station_idx: int, to_station_idx: int, number: int = 1):
        self.from_station_idx = from_station_idx
        self.to_station_idx = to_station_idx
        self.number = number

    def __repr__(self):
        return "%s {from_station_idx: %r, to_station_idx: %r, number:%r}" % \
            (self.__class__.__name__, self.from_station_idx, self.to_station_idx, self.number)


class DecisionType(Enum):
    """Station decision type."""
    # current cell has too more bikes, need transfer to others
    Supply = 'supply'
    # current cell has no enough bikes, need neighbors transfer bikes to it
    Demand = 'demand'


class DecisionEvent:
    """Citi bike scenario decision event that contains station information for agent to choose action.

    Args:
        station_idx (int): Which station need an action.
        tick (int): Current simulator tick.
        frame_index (int): Frame index of current tick, used to query from snapshots.
        action_scope_func (callable): Function to retrieve latest action scope states.
        decision_type (DecisionType): The type of this decision.
    """

    summary_key = ["station_idx", "tick", "frame_index", "type", "action_scope"]

    def __init__(
        self, station_idx: int, tick: int, frame_index: int, action_scope_func: callable, decision_type: DecisionType
    ):
        self.station_idx = station_idx
        self.tick = tick
        self.frame_index = frame_index
        self.type = decision_type
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self) -> dict:
        """dict: A dictionary that contains requirements of current and neighbor stations,
                key is the station index, value is the max demand or supply number.
        """
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.station_idx, self.type)

        return self._action_scope

    def __getstate__(self):
        """Return pickleable dictionary."""
        return {
            "station_idx": self.station_idx,
            "tick": self.tick,
            "frame_index": self.frame_index,
            "type": self.type,
            "action_scope": self.action_scope}

    def __setstate__(self, state):
        self.station_idx = state["station_idx"]
        self.tick = state["tick"]
        self.frame_index = state["frame_index"]
        self.type = state["type"]
        self._action_scope = state["action_scope"]

    def __repr__(self):
        return "%s {station_idx: %r, type: %r, action_scope:%r}" % \
            (self.__class__.__name__, self.station_idx, str(self.type), self.action_scope)


class Action:
    """Citi bike scenario action object, that used to pass action from agent to business engine.

    Args:
        from_station_idx (int): Which station will take this acion.
        to_station_idx (int): Which station is the target of this action.
        number (int): Bike number to transfer.
    """

    def __init__(self, from_station_idx: int, to_station_idx: int, number: int):
        self.from_station_idx = from_station_idx
        self.to_station_idx = to_station_idx
        self.number = number

    def __repr__(self):
        return "%s {from_station_idx: %r, to_station_idx: %r, number:%r}" % \
            (self.__class__.__name__, self.from_station_idx, str(self.to_station_idx), self.number)


class ExtraCostMode(Enum):
    """The mode to process extra cost."""
    Source = "source"
    Target = "target"
    # TargetNeighbors = "target_neighbors"
