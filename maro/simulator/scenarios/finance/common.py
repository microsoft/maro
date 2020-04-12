from enum import Enum
from typing import Dict, List, Callable

from .abs_sub_business_engine import AbsSubBusinessEngine


class FinanceType(Enum):
    stock = "stock",
    futures = "futures"

class DecisionEvent:
    def __init__(self, tick: int, type: FinanceType, items: list, sub_engine_name: str, action_scope_func: Callable):
        """
        Parameters:
            tick (int): current tick of decision
            type (FinanceType): type of this decision
            items (list): list of available items for actions, such as stock for StockSubEngine
            sub_engine_name (str): name of sub-engine, used to identify which this decision come from, as we support multi-sub-engine with same type, we need this field from action
            action_scope_func (Callable): function to provide action scope
        """
        self.tick = tick
        self.type = type
        self.items = items
        self.sub_engine_name = sub_engine_name
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self) -> list:
        """Action scope for items"""
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.items)
        
        return self._action_scope

    def __repr__(self):
        return f"<DecisionEvent type: {self.type}, tick: {self.tick}, engine: {self.sub_engine_name} >"


class Action:
    def __init__(self, sub_engine_name: str, item_index: int, number: int):
        """
        Parameters:
            sub_engine_name (str): name of engine the decision event from
            item_index (int): index of the item (such as stock index), usually from DecisionEvent.items
            number (int): number to perform, positive means buy, negitive means sell
        """
        self.sub_engine_name = sub_engine_name
        self.item_index = item_index
        self.number = number

    def __repr__(self):
        return f"<Action engine: {self.sub_engine_name} item: {self.item_index} number: {self.number}>"


class SubEngineAccessWrapper:
    class PropertyAccessor:
        def __init__(self, properties: dict):
            self._properties = properties

        def __getitem__(self, name: str):
            """Used to access frame/snapshotlist by name as a dictionary."""
            if name not in self._properties:
                return None
            
            return self._properties[name]

        def __getattribute__(self, name):
            """Used to access frame/snapshotlist by name as a attribute"""
            # if name in self._properties:
                # return self._properties[name]
            properties = object.__getattribute__(self, "_properties")

            return properties[name]

    """Wrapper to access frame/config/snapshotlist by name of sub-engine"""
    def __init__(self, sub_engines: dict):
        self._engines = sub_engines

    def get_property_access(self, property_name: str):
        properties = {name: getattr(engine, property_name) for name, engine in self._engines.items()}

        return SubEngineAccessWrapper.PropertyAccessor(properties)
