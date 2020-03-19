from abc import ABC, abstractmethod
from maro.simulator.graph import SnapshotList 

class AbsDecisionStrategy:
    def __init__(self, stations: list, options: dict):
        self._stations = stations
        self._options = options
    
    @abstractmethod
    def get_stations_need_decision(self, tick: int, unit_tick: int) -> list:
        """Return a list of station index which need to take an action"""
        pass