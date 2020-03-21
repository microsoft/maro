class BikeDecisionStrategy:
    def __init__(self, cells: list, options: dict):
        self._cells = cells
        self._options = options


    def get_cells_need_decision(self, tick: int, unit_tick: int) -> list:
        """Get cells that need to take an action from agent at current tick"""
        cells = []

        # for cell in self._cells:
        #     if cell.bikes/cell.capacity <= self._threshold:
        #         ret.append(cell.index)

        return cells

    def action_scope(self, cell_idx: int):
        """Calculate action scope base on config"""
        pass