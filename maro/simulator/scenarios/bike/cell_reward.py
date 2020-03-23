from .cell import Cell

class CellReward:
    def __init__(self, cells: list, options: dict):
        self._cells = cells
        self._alpha = options["alpha"]
        self._beta = options["beta"]
        self._gamma = options["gamma"]

    def reward(self, cell_idx: int):
        
        cell: Cell = self._cells[cell_idx]

        reward = self._alpha * cell.shortage + self._beta * cell.fulfillment + self._gamma * cell.trip_requirement

        return reward