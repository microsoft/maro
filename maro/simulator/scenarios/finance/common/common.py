from enum import Enum
from typing import Callable
from abc import ABC, abstractmethod


class OrderMode(Enum):
    market_order = "market_order"
    limit_order = "limit_order"
    stop_order = "stop_order"
    stop_limit_order = "stop_limit_order"
    cancel_order = "cancel_order"


class OrderDirection(Enum):
    buy = "buy"
    sell = "sell"


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
            self, trade_number: int,
            price_per_item: float, tax: float
        ):
        self.trade_number = trade_number
        self.price_per_item = price_per_item
        self.tax = tax

    @property
    def total_cost(self):
        return self.trade_number * self.price_per_item + self.tax

    def __repr__(self):
        return f"<  \
            number: {self.trade_number} price: {self.price_per_item} \
            tax: {self.tax} >"


class DecisionEvent:
    def __init__(
        self, tick: int, item: int = -1,
        action_scope_func: Callable = None, action_type: ActionType = ActionType.order
    ):
        """
        Parameters:
            tick (int): current tick of decision
            item (int): available item index for action, such as a stock for StockSubEngine
            action_scope_func (Callable): function to provide action scope
            action_type (ActionType): type of the expected action
        """
        self.tick = tick
        self.action_type = action_type
        self.item = item
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self):
        """Action scope for items"""
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.action_type, self.item, self.tick)

        return self._action_scope

    def __repr__(self):
        return f"<DecisionEvent type: {self.action_type}, tick: {self.tick}>"

class OrderActionScope:
    def __init__(self, buy_min, buy_max, sell_min, sell_max, supported_order):
        self.buy_min = buy_min
        self.buy_max = buy_max
        self.sell_min = sell_min
        self.sell_max = sell_max
        self.supported_order = supported_order

class CancelOrderActionScope:
    def __init__(self, available_orders):
        self.available_orders = available_orders

class Action(ABC):
    idx = 0

    def __init__(
        self,
        tick: int = 0,
        id: int = None, life_time: int = 1
    ):
        """
        Parameters:
            item_index (int): index of the item (such as stock index), usually from DecisionEvent.items
            number (int): number to perform, positive means buy, negitive means sell
        """
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
        self.comment = ""
        # print("Action id:", self.id)

    def __repr__(self):
        return f"< Action decision: {self.decision_tick} finished: {self.finish_tick} state: {self.state} >"


class CancelOrder(Action):
    def __init__(self, action, tick):
        super().__init__(
            tick=tick, life_time=1
        )
        self.action = action


class Order(Action):
    def __init__(
        self, item: int, amount: int, mode: OrderMode,
        direction: OrderDirection, tick: int, life_time: int = 1
    ):
        super().__init__(
            tick=tick, life_time=life_time
        )
        self.item = item
        self.amount = amount
        self.mode = mode
        self.direction = direction
    
    @abstractmethod
    def is_trigger(self, price, trade_volume) -> bool:
        pass


class MarketOrder(Order):
    def __init__(
        self, tick: int, item: int, amount: int, direction: OrderDirection, life_time: int = 1):
        super().__init__(
            item=item, amount=amount, direction=direction,
            mode=OrderMode.market_order, tick=tick, life_time=life_time
        )

    def is_trigger(self, price, trade_volume) -> bool:
        triggered = False
        if trade_volume > 0:
            triggered = True
        # print(f'Market Order triggered: {triggered}')
        return triggered


class LimitOrder(Order):
    def __init__(
        self, tick: int, item: int, amount: int, direction: OrderDirection,
        limit: float, life_time: int = 1
    ):
        super().__init__(
            item=item, amount=amount, direction=direction,
            mode=OrderMode.limit_order, tick=tick, life_time=life_time
        )
        self.limit = limit

    def is_trigger(self, price, trade_volume) -> bool:
        triggered = False
        if trade_volume > 0:
            if self.direction == OrderDirection.buy:  # buy
                if self.limit >= price:
                    triggered = True
            else:  # sell
                if self.limit <= price:
                    triggered = True
        # print(f'Limit Order triggered: {triggered}')
        return triggered


class StopOrder(Order):
    def __init__(
        self, tick: int, item: int, amount: int, direction: OrderDirection,
        stop: float, life_time: int = 1
    ):
        super().__init__(
            item=item, amount=amount, direction=direction,
            mode=OrderMode.stop_order, tick=tick, life_time=life_time
        )
        self.stop = stop
    
    def is_trigger(self, price, trade_volume) -> bool:
        triggered = False
        if trade_volume > 0:
            if self.direction == OrderDirection.buy:  # buy
                if self.stop <= price:
                    triggered = True
            else:  # sell
                if self.stop >= price:
                    triggered = True
        # print(f'Stop Order triggered: {triggered}')
        return triggered


class StopLimitOrder(Order):
    def __init__(
        self, tick: int, item: int, amount: int, direction: OrderDirection,
        stop: float, limit: float, life_time: int = 1
    ):
        super().__init__(
            item=item, amount=amount, direction=OrderDirection,
            mode=OrderMode.stop_limit_order, tick=tick, life_time=life_time
        )
        self.stop = stop
        self.limit = limit
    
    def is_trigger(self, price, trade_volume) -> bool:
        triggered = False
        if trade_volume > 0:
            if self.direction == OrderDirection.buy:  # buy
                if self.stop <= price:
                    if self.limit >= price:
                        triggered = True
            else:  # sell
                if self.stop >= price:
                    if self.limit <= price:
                        triggered = True

        # print(f'Stop Limit Order triggered: {triggered}')
        return triggered