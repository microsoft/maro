from .cell import Cell
from math import floor
import numpy as np

class BikeDecisionStrategy:
    def __init__(self, cells: list, options: dict):
        self._cells = cells

        self.resolution = options["resolution"]
        self.time_mean = options["effective_time_mean"]
        self.high_water_mark_ratio = options["high_water_mark_ratio"]
        self.low_water_mark_ratio = options["low_water_mark_ratio"]
        self.time_std = options["effective_time_std"]

    def get_cells_need_decision(self, tick: int, internal_tick: int) -> list:
        """Get cells that need to take an action from agent at current tick"""

        cells = []

        if (internal_tick + 1) % self.resolution == 0:
            for cell in self._cells:
                # if cell has too many available bikes, then we ask an action
                if cell.bikes/cell.capacity >= self.high_water_mark_ratio:
                    cells.append(cell.index)

        return cells

    def action_scope(self, cell_idx: int):
        """Calculate action scope base on config"""
        cell: Cell = self._cells[cell_idx]
        neighbor_num = len(cell.neighbors)
        scope = {}

        # how many bikes we can supply to other cells from current cell
        scope[cell_idx] = floor(cell.bikes * (1 - self.low_water_mark_ratio))

        for neighbor_idx in cell.neighbors:
            if neighbor_idx >=0:
                neighbor_cell = self._cells[neighbor_idx]

                # we should not transfer bikes to a cell which already meet the high water mark ratio
                max_bikes = floor(neighbor_cell.capacity * self.high_water_mark_ratio)
                scope[neighbor_idx] = max(max_bikes - neighbor_cell.bikes, 0)
        
        return scope

    @property
    def transfer_time(self):
        """Transfer time from one cell to another"""
        return round(np.random.normal(self.time_mean, scale=self.time_std))