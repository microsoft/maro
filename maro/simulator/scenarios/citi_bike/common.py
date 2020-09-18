# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.s

from enum import Enum
from datetime import datetime

class BikeTransferPayload:
    def __init__(self, from_station_idx: int, to_station_idx: int, number: int=1):
        self.from_station_idx = from_station_idx
        self.to_station_idx = to_station_idx
        self.number = number


class BikeReturnPayload:
    def __init__(self, from_station_idx: int, to_station_idx: int, number: int = 1):
        self.from_station_idx = from_station_idx
        self.to_station_idx = to_station_idx
        self.number = number


class DecisionType(Enum):
    Supply = 'supply' # current cell has too more bikes, need transfer to others
    Demand = 'demand' # current cell has no enough bikes, need neighbors transfer bikes to it


class DecisionEvent:
    def __init__(self, station_idx: int, tick: int, frame_index: int, action_scope_func: callable, decision_type: DecisionType):
        self.station_idx = station_idx
        self.tick = tick
        self.frame_index = frame_index
        self.type = decision_type
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self):
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.station_idx, self.type)

        return self._action_scope

    def __getstate__(self):
        """Return pickleable dictionary.
        
        NOTE: this class do not support unpickle"""
        return {
            "station_idx": self.station_idx,
            "tick": self.tick,
            "frame_index": self.frame_index,
            "type": self.type,
            "action_scope": self.action_scope}

    def __repr__(self):
        return f"decision event {self.__getstate__()}"


class Action:
    def __init__(self, from_station_idx: int, to_station_idx: int, number: int):
        self.from_station_idx = from_station_idx
        self.to_station_idx = to_station_idx
        self.number = number


class ExtraCostMode(Enum):
    Source = "source"
    Target = "target"
    # TargetNeighbors = "target_neighbors"