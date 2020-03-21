from .cell import Cell
from math import floor
import numpy as np

class BikeDecisionStrategy:
    def __init__(self, cells: list, options: dict):
        self._cells = cells

        self.resolution = options["resolution"]
        self.std = options["std"]
        self.high = options["high"]
        self.low = options["low"]
        self.scale = options["scale"]

    def get_cells_need_decision(self, tick: int, internal_tick: int) -> list:
        """Get cells that need to take an action from agent at current tick"""

        cells = []

        if (internal_tick + 1) % self.resolution == 0:
            for cell in self._cells:
                # if cell has too many available bikes, then we ask an action
                if cell.bikes/cell.capacity >= self.high:
                    cells.append(cell.index)

        return cells

    def action_scope(self, cell_idx: int):
        """Calculate action scope base on config"""
        cell: Cell = self._cells[cell_idx]

        # how many bikes we can supply to other cells
        return floor(cell.bikes * (1-self.low))

    @property
    def transfer_time(self):
        """Transfer time from one cell to another"""
        return round(np.random.normal(self.std, scale=self.scale))