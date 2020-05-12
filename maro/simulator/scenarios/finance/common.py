from enum import Enum
from typing import Dict, List, Callable
from inspect import isfunction
from functools import partial
from maro.simulator.utils.common import BaseAction


class FinanceType(Enum):
    stock = "stock",
    futures = "futures"


class OrderMode(Enum):
    market_order = "market_order"
    limit_order = "limit_order"
    stop_order = "stop_order"
    stop_limit_order = "stop_limit_order"


class TradeResult:
    """Result or a trade order"""

    def __init__(self, sub_engine_name: str, item_index: int, trade_number: int, tick: int, price_per_item: float, tax: float, is_trade_accept: bool = False):
        self.sub_engine_name = sub_engine_name
        self.item_index = item_index
        self.trade_number = trade_number
        self.tick = tick
        self.price_per_item = price_per_item
        self.tax = tax
        self.is_trade_accept = is_trade_accept

    @property
    def total_cost(self):
        return self.trade_number * self.price_per_item + self.tax

    def __repr__(self):
        return f"< trade sub-engine: {self.sub_engine_name} item: {self.item_index} tick: {self.tick} number: {self.trade_number} price: {self.price_per_item} tax: {self.tax} tradable: {self.is_trade_accept} >"


class DecisionEvent:
    def __init__(self, tick: int, type: FinanceType, item: int, sub_engine_name: str, action_scope_func: Callable):
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
        self.item = item
        self.sub_engine_name = sub_engine_name
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self) -> tuple:
        """Action scope for items"""
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.item)

        return self._action_scope

    def __repr__(self):
        return f"<DecisionEvent type: {self.type}, tick: {self.tick}, engine: {self.sub_engine_name} >"


class Action(BaseAction):
    def __init__(self, sub_engine_name: str, item_index: int, number: int, order_mode: OrderMode = None, stop: int = 0, limit: int = 0):
        """
        Parameters:
            sub_engine_name (str): name of engine the decision event from
            item_index (int): index of the item (such as stock index), usually from DecisionEvent.items
            number (int): number to perform, positive means buy, negitive means sell
        """
        super().__init__()
        self.sub_engine_name = sub_engine_name
        self.item_index = item_index
        self.number = number
        self.order_mode = order_mode
        self.stop = stop
        self.limit = limit

    def __repr__(self):
        return f"<Action engine: {self.sub_engine_name} item: {self.item_index} number: {self.number}>"


class SubEngineAccessWrapper:
    class PropertyAccessor:
        def __init__(self, properties: dict):
            self._properties = properties
            self._customized = {}

        def add_item(self, key, val):
            self._customized[key] = val

        def __getitem__(self, name: str):
            """Used to access frame/snapshotlist by name as a dictionary."""
            if name in self._properties:
                return self._properties[name]
            elif name in self._customized:
                return self._customized[name]

            return None

        def __getattribute__(self, name):
            properties = object.__getattribute__(self, "_properties")
            customized = object.__getattribute__(self, "_customized")

            if name in properties or name in customized:
                # used to access frame/snapshotlist by name as an attribute, such as env.snapshotlist.sub_a.xxxx
                return properties[name] if name in properties else customized[name]
            else:
                # used to compact with current core to insert snapshot, or other implementation

                # HOTFIX for special cases
                if name == "insert_snapshot":
                    return partial(SubEngineAccessWrapper.insert_snapshot, self)

            return super().__getattribute__(name)

    """Wrapper to access frame/config/snapshotlist by name of sub-engine"""

    def __init__(self, sub_engines: dict):
        self._engines = sub_engines

    def get_property_access(self, property_name: str):
        properties = {name: getattr(engine, property_name) for name, engine in self._engines.items()}

        return SubEngineAccessWrapper.PropertyAccessor(properties)

    @staticmethod
    def insert_snapshot(snapshot_list_acc: PropertyAccessor, index):
        """Used to support insert snapshot for property accessor"""
        for name, snapshot_list in snapshot_list_acc._properties.items():
            snapshot_list.insert_snapshot(index)
