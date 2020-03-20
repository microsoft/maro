from abc import ABC, abstractmethod
from maro.simulator.graph import SnapshotList 

class AbsDecisionStrategy:
    def __init__(self, cells: list, options: dict):
        self._cells = cells
        self._options = options
    
    @abstractmethod
    def get_cells_need_decision(self, tick: int, unit_tick: int) -> list:
        """Return a list of cell index which need to take an action"""
        pass