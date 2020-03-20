

class Order:
    def __init__(self, date, start_cell: int, end_cell: int, end_tick: int, usertype: int = 0, gendor:int = 0, num: int=1):
        self.date = date
        self.start_cell = start_cell
        self.end_cell = end_cell
        self.number = num
        self.end_tick = end_tick
        self.usertype = usertype
        self.gendor = gendor

    @property
    def weekday(self):
        return self.date.weekday()

    def __repr__(self):
        return f"(Order start cell: {self.start_cell}, end cell: {self.end_cell}, end tick: {self.end_tick})"

class BikeTransferPaylod:
    def __init__(self, from_cell:int, to_cell: int, number:int):
        self.from_cell = from_cell
        self.to_cell = to_cell
        self.number = number

class BikeReturnPayload:
    def __init__(self, target_cell_idx: int, num: int = 1):
        self.target_cell = target_cell_idx
        self.number = num

    def __repr__(self):
        return f"(Bike return payload, target cell: {self.target_cell}, number: {self.number})"


class DecisionEvent:
    def __init__(self, cell_idx: int, tick: int, action_scope_func: callable):
        self.cell_idx = cell_idx
        self.tick = tick
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self):
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.cell_idx)

        return self._action_scope

    def __repr__(self):
        return f"DecisionEvent(cell: {self.cell_idx}, action scope: {self.action_scope})"

class Action:
    def __init__(self, cell_idx:int, to_cell: int, number: int):
        self.cell = cell_idx
        self.to_cell = to_cell
        self.number = number

    def __repr__(self):
        return f"Action(cell: {self.cell}, number: {self.number})"