from typing import Callable
from maro.simulator.frame import SnapshotList



class Demand:
    def __init__(self, demand: int):
        self.demand = demand

    def __repr__(self):
        return f"(Demand value: {self.demand})"


class Action:
    def __init__(self, warehouse_idx: int, restock: int):
        self.warehouse_idx = warehouse_idx
        self.restock = restock

    def __repr__(self):
        return f"Action(restock: {self.restock})"


class ActionScope:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'ActionScope'


class DecisionEvent:
    def __init__(self, tick:int, warehouse_idx:int, snapshot_list: SnapshotList, action_scope_func: Callable):
        self.tick = tick
        self.warehouse_idx = warehouse_idx
        self.snapshot_list = snapshot_list
        
        self._action_scope_func = action_scope_func
        self._action_scope = None

    @property
    def action_scope(self):
        # NOTE: we use a function call instead of value to make sure the agent can always get latest states
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.warehouse_idx)
        
        return self._action_scope