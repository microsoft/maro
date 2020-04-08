from .cell import Cell
from math import floor
from .common import DecisionType
import numpy as np

class BikeDecisionStrategy:
    def __init__(self, cells: list, options: dict):
        self._cells = cells

        self.resolution = options["resolution"]
        self.time_mean = options["effective_time_mean"]
        self.high_water_mark_ratio = options["high_water_mark_ratio"]
        self.low_water_mark_ratio = options["low_water_mark_ratio"]
        self.time_std = options["effective_time_std"]

        
        self.scope_low_ratio = 0 #Eoptions["action_scope"]["low"]
        self.scope_high_ratio = 1 #options["action_scope"]["high"]

        if "action_scope" in options:
            self.scope_low_ratio = options["action_scope"]["low"]
            self.scope_high_ratio = options["action_scope"]["high"]

    def get_cells_need_decision(self, tick: int) -> list:
        """Get cells that need to take an action from agent at current tick"""

        cells = []

        if (tick + 1) % self.resolution == 0:
            for cell in self._cells:
                cur_ratio = cell.bikes / cell.capacity
                # if cell has too many available bikes, then we ask an action
                if cur_ratio >= self.high_water_mark_ratio:
                    cells.append((cell.index, DecisionType.Supply))
                elif cur_ratio <= self.low_water_mark_ratio:
                    cells.append((cell.index, DecisionType.Demand))

        return cells

    def action_scope(self, cell_idx: int, decision_type: DecisionType):
        """Calculate action scope base on config"""
        cell: Cell = self._cells[cell_idx]
        neighbor_num = len(cell.neighbors)
        scope = {}

        # how many bikes we can supply to other cells from current cell
        if decision_type == DecisionType.Supply:
            scope[cell_idx] = floor(cell.bikes * (1 - self.scope_low_ratio))
        else:
            # how many bike we can accept
            scope[cell_idx] = cell.capacity - cell.bikes

        for neighbor_idx in cell.neighbors:
            if neighbor_idx >=0:
                neighbor_cell: Cell = self._cells[neighbor_idx]

                # we should not transfer bikes to a cell which already meet the high water mark ratio
                if decision_type == DecisionType.Supply:
                    # for supply decision, we provide max bikes that neighbor can accept
                    max_bikes = neighbor_cell.capacity - neighbor_cell.bikes
                else:
                    # for demand decision, this will be max bikes that neighbor can provide
                    max_bikes = floor(neighbor_cell.bikes * self.scope_high_ratio)

                scope[neighbor_idx] = max_bikes
        
        return scope

    @property
    def transfer_time(self):
        """Transfer time from one cell to another"""
        return round(np.random.normal(self.time_mean, scale=self.time_std))