import calendar
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

from maro.data_lib.binary_converter import is_datetime
from maro.data_lib.binary_reader import unit_seconds


class OrderMode(Enum):
    MARKET_ORDER = "market_order"
    LIMIT_ORDER = "limit_order"
    STOP_ORDER = "stop_order"
    STOP_LIMIT_ORDER = "stop_limit_order"


class OrderDirection(Enum):
    BUY = "buy"
    SELL = "sell"


class ActionType(Enum):
    ORDER = "order"
    CANCEL_ORDER = "cancel_order"


class ActionState(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELED = "canceled"


class FinanceEventType(Enum):
    """Event type for CIM problem."""
    # RELEASE_EMPTY = 10
    ORDER = 11
    CANCEL_ORDER = 12


class TradeResult:
    """Result or a trade order"""

    def __init__(
            self, trade_direction: OrderDirection, trade_volume: int, trade_price: float,
            commission: float, tax: float
    ):
        self.trade_volume = int(trade_volume)
        self.trade_price = trade_price
        self.commission = commission
        self.tax = tax
        self.trade_direction = trade_direction

    @property
    def cash_delta(self):
        return (self.trade_volume * self.trade_price - self.commission - self.tax) * \
            (-1 if self.trade_direction == OrderDirection.BUY else 1)

    def __repr__(self):
        return f"<  \
            number: {self.trade_volume} price: {self.trade_price} \
            commission: {self.commission} tax: {self.tax} >"


class DecisionEvent:
    def __init__(
        self, tick: int, stock_index: int = -1,
        action_scope_func: Callable = None, action_type: ActionType = ActionType.ORDER
    ):
        """
        Args:
            tick (int): current tick of decision
            stock_index (int): available item index for action, such as a stock for StockSubEngine
            action_scope_func (Callable): function to provide action scope
            action_type (ActionType): type of the expected action
        """
        self.tick = tick
        self.action_type = action_type
        self.stock_index = stock_index
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self):
        """Action scope for items"""
        if self._action_scope is None:
            self._action_scope = self._action_scope_func(self.action_type, self.stock_index, self.tick)

        return self._action_scope

    def __repr__(self):
        return f"<DecisionEvent type: {self.action_type}, tick: {self.tick}, action scope: {self.action_scope}>"


class OrderActionScope:
    def __init__(self, min_buy_volume, max_buy_volume, min_sell_volume, max_sell_volume, supported_order_mode):
        self.min_buy_volume = min_buy_volume
        self.max_buy_volume = max_buy_volume
        self.min_sell_volume = min_sell_volume
        self.max_sell_volume = max_sell_volume
        self.supported_order_mode = supported_order_mode

    def __repr__(self):
        return f"buy scope: {self.min_buy_volume}~{self.max_buy_volume} \
sell scope: {self.min_sell_volume}~{self.max_sell_volume} order types: {self.supported_order_mode}"


class CancelOrderActionScope:
    def __init__(self, available_orders):
        self.available_orders = available_orders

    def __repr__(self):
        return f"available orders: {self.available_orders}"


class Action(ABC):
    idx = 0

    def __init__(
        self,
        tick: int = 0,
        id: int = None, life_time: int = 1
    ):
        """
        Args:
            item_index (int): index of the item (such as stock index), usually from DecisionEvent.stock_indexs
            number (int): number to perform, positive means buy, negitive means sell
        """
        self.decision_tick = tick
        self.finish_tick = None
        self.state = ActionState.PENDING
        if id is not None:
            self.id = id
        else:
            self.id = Action.idx
            Action.idx += 1
        self.remaining_life_time = life_time
        self.action_result = None
        self.comment = ""
        # print("Action id:", self.id)

    def __repr__(self):
        return f"< Action ID:{self.id} start: {self.decision_tick} finished: {self.finish_tick} state: {self.state} \
remaining life time:{self.remaining_life_time} comment: {self.comment}>"


class Order(Action):
    def __init__(
        self, stock_index: int, order_volume: int, order_mode: OrderMode,
        order_direction: OrderDirection, tick: int, life_time: int = 1
    ):
        super().__init__(
            tick=tick, life_time=life_time
        )
        self.stock_index = stock_index
        self.order_volume = order_volume
        self.order_mode = order_mode
        self.order_direction = order_direction

    @abstractmethod
    def is_trigger(self, price, market_volume) -> bool:
        pass

    def __repr__(self):
        return f"{super().__repr__()}\n< Order item: {self.stock_index} volume: {self.order_volume} \
direction: {self.order_direction} >"


class MarketOrder(Order):
    def __init__(
        self, tick: int, stock_index: int, order_volume: int, order_direction: OrderDirection, life_time: int = 1
    ):
        super().__init__(
            stock_index=stock_index, order_volume=order_volume, order_direction=order_direction,
            order_mode=OrderMode.MARKET_ORDER, tick=tick, life_time=life_time
        )

    def is_trigger(self, price, market_volume) -> bool:
        triggered = False
        if market_volume > 0:
            triggered = True
        # print(f'Market Order triggered: {triggered}')
        return triggered


class LimitOrder(Order):
    def __init__(
        self, tick: int, stock_index: int, order_volume: int, order_direction: OrderDirection,
        limit: float, life_time: int = 1
    ):
        super().__init__(
            stock_index=stock_index, order_volume=order_volume, order_direction=order_direction,
            order_mode=OrderMode.LIMIT_ORDER, tick=tick, life_time=life_time
        )
        self.limit = limit

    def is_trigger(self, price, market_volume) -> bool:
        triggered = False
        if market_volume > 0:
            if self.order_direction == OrderDirection.BUY:  # buy
                if self.limit >= price:
                    triggered = True
            else:  # sell
                if self.limit <= price:
                    triggered = True
        # print(f'Limit Order triggered: {triggered}')
        return triggered


class StopOrder(Order):
    def __init__(
        self, tick: int, stock_index: int, order_volume: int, order_direction: OrderDirection,
        stop: float, life_time: int = 1
    ):
        super().__init__(
            stock_index=stock_index, order_volume=order_volume, order_direction=order_direction,
            order_mode=OrderMode.STOP_ORDER, tick=tick, life_time=life_time
        )
        self.stop = stop

    def is_trigger(self, price, market_volume) -> bool:
        triggered = False
        if market_volume > 0:
            if self.order_direction == OrderDirection.BUY:  # buy
                if self.stop <= price:
                    triggered = True
            else:  # sell
                if self.stop >= price:
                    triggered = True
        # print(f'Stop Order triggered: {triggered}')
        return triggered


class StopLimitOrder(Order):
    def __init__(
        self, tick: int, stock_index: int, order_volume: int, order_direction: OrderDirection,
        stop: float, limit: float, life_time: int = 1
    ):
        super().__init__(
            stock_index=stock_index, order_volume=order_volume, order_direction=order_direction,
            order_mode=OrderMode.STOP_LIMIT_ORDER, tick=tick, life_time=life_time
        )
        self.stop = stop
        self.limit = limit

    def is_trigger(self, price, market_volume) -> bool:
        triggered = False
        if market_volume > 0:
            if self.order_direction == OrderDirection.BUY:  # buy
                if self.stop <= price:
                    if self.limit >= price:
                        triggered = True
            else:  # sell
                if self.stop >= price:
                    if self.limit <= price:
                        triggered = True

        # print(f'Stop Limit Order triggered: {triggered}')
        return triggered


class CancelOrder(Action):
    def __init__(self, order: Order, tick: int):
        super().__init__(
            tick=tick, life_time=1
        )
        self.order = order


def two_decimal_price(input_price: float) -> float:
    return int(input_price * 100) / 100


def get_cn_stock_data_tick(start_date: str) -> int:
    ret = None
    tzone = "Asia/Shanghai"
    default_start_dt = "1991-01-01"
    default_time_unit = "d"
    is_dt, dt = is_datetime(start_date, tzone)
    if is_dt:
        # convert into UTC, then utc timestamp
        # dt = dt.astimezone(UTC)
        _, start_dt = is_datetime(default_start_dt, tzone)
        dt_seconds = calendar.timegm(dt.timetuple())
        start_dt_seconds = calendar.timegm(start_dt.timetuple())
        delta_seconds = dt_seconds - start_dt_seconds
        seconds_per_unit = unit_seconds(default_time_unit)
        ret = int((delta_seconds) / seconds_per_unit)
    return ret


def get_stock_start_timestamp(start_date: str = "1991-01-01", tzone: str = "Asia/Shanghai") -> int:
    ret = None
    default_start_dt = "1970-01-01"
    default_start_tzone = "UTC"
    default_time_unit = "s"
    is_dt, dt = is_datetime(start_date, tzone)
    if is_dt:
        # convert into UTC, then utc timestamp
        # dt = dt.astimezone(UTC)
        _, start_dt = is_datetime(default_start_dt, default_start_tzone)
        # start_dt = start_dt.astimezone(UTC)
        dt_seconds = calendar.timegm(dt.timetuple())
        start_dt_seconds = calendar.timegm(start_dt.timetuple())
        delta_seconds = dt_seconds - start_dt_seconds
        seconds_per_unit = unit_seconds(default_time_unit)
        ret = int(delta_seconds / seconds_per_unit)
    return ret
