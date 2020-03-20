from .abs_decisionstrategy import AbsDecisionStrategy


class BikePercentDecisionStrategy(AbsDecisionStrategy):
    """cells will trigger a decision event when its available bikes touch a percentage bar"""
    def __init__(self, cells: list, options: dict):
        super().__init__(cells, options)

        # this strategy need a percentage config
        self._threshold = options["threshold"]


    def get_cells_need_decision(self, tick: int, unit_tick: int) -> list:
        ret = []

        for cell in self._cells:
            if cell.bikes/cell.capacity <= self._threshold:
                ret.append(cell.index)

        return ret