from enum import Enum
from typing import Callable
from functools import partial


class FinanceType(Enum):
    stock = "stock",
    futures = "futures"


class OrderMode(Enum):
    market_order = "market_order"
    limit_order = "limit_order"
    stop_order = "stop_order"
    stop_limit_order = "stop_limit_order"
    cancel_order = "cancel_order"


class ActionType(Enum):
    order = "order"
    cancel_order = "cancel_order"


class ActionState(Enum):
    pending = "pending"
    success = "success"
    failed = "failed"
    expired = "expired"
    canceled = "canceled"


class TradeResult:
    """Result or a trade order"""

    def __init__(
            self, item_index: int, trade_number: int, tick: int,
            price_per_item: float, tax: float, action_id: int, is_trade_accept: bool = False,
            is_trade_trigger: bool = True):
        self.item_index = item_index
        self.trade_number = trade_number
        self.tick = tick
        self.price_per_item = price_per_item
        self.tax = tax
        self.is_trade_accept = is_trade_accept
        self.is_trade_trigger = is_trade_trigger
        self.action_id = action_id

    @property
    def total_cost(self):
        return self.trade_number * self.price_per_item + self.tax

    def __repr__(self):
        return f"< trade item: {self.item_index} \
            tick: {self.tick} number: {self.trade_number} price: {self.price_per_item} \
            tax: {self.tax} tradable: {self.is_trade_accept} >"


class DecisionEvent:
    def __init__(
        self, tick: int, type: FinanceType, item: int,
        action_scope_func: Callable, action_type: ActionType
    ):
        """
        Parameters:
            tick (int): current tick of decision
            type (FinanceType): type of this decision
            item (int): available item index for action, such as a stock for StockSubEngine
            sub_engine_name (str): name of sub-engine, used to identify which this decision come from,
                as we support multi-sub-engine with same type, we need this field from action
            action_scope_func (Callable): function to provide action scope
        """
        self.tick = tick
        self.type = type
        self.action_type = action_type
        self.item = item
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self) -> list:
        """Action scope for items"""
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.action_type, self.item)

        return self._action_scope

    def __repr__(self):
        return f"<DecisionEvent type: {self.type}, tick: {self.tick}>"


class Action:
    idx = 0

    def __init__(
        self, item_index: int = 0, number: int = 0,
        action_type: ActionType = ActionType.order, tick: int = 0, order_mode: OrderMode = None,
        stop: float = 0, limit: float = 0, id: int = None, life_time: int = 1
    ):
        """
        Parameters:
            sub_engine_name (str): name of engine the decision event from
            item_index (int): index of the item (such as stock index), usually from DecisionEvent.items
            number (int): number to perform, positive means buy, negitive means sell
        """
        self.item_index = item_index
        self.number = number
        self.action_type = action_type
        self.order_mode = order_mode
        self.stop = stop
        self.limit = limit
        self.decision_tick = tick
        self.finish_tick = None
        self.state = ActionState.pending
        if id is not None:
            self.id = id
        else:
            self.id = Action.idx
            Action.idx += 1
        self.life_time = life_time
        self.action_result = None
        print("Action id:", self.id)

    def __repr__(self):
        return f"< Action item: {self.item_index} number: {self.number} action type: \
            {self.action_type} decision: {self.decision_tick} finished: {self.finish_tick} >"
