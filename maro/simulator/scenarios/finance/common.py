import calendar
from abc import ABC, abstractmethod
from enum import Enum

from maro.data_lib.binary_converter import is_datetime
from maro.data_lib.binary_reader import unit_seconds


class FinanceEventType(Enum):
    """Type of the event in Finance scenario."""
    ORDER = "order"
    CANCEL = "cancel"


class OrderMode(Enum):
    """Mode of the order, ref to https://www.investopedia.com/terms/o/order.asp ."""
    MARKET_ORDER = "market_order"
    LIMIT_ORDER = "limit_order"
    STOP_ORDER = "stop_order"
    STOP_LIMIT_ORDER = "stop_limit_order"


class OrderDirection(Enum):
    """Direction of the order."""
    BUY = "buy"
    SELL = "sell"


class ActionState(Enum):
    """State of the action."""
    REQUEST = "request"
    PENDING = "pending"
    FINISH = "finish"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCEL = "cancel"


class Trade:
    """Class for a trade."""

    def __init__(
            self, trade_direction: OrderDirection, trade_volume: int, trade_price: float,
            trade_cost: float
    ):
        self.trade_volume = int(trade_volume)
        self.trade_price = trade_price
        self.trade_cost = trade_cost
        self.trade_direction = trade_direction

    @property
    def cash_delta(self):
        delta = 0
        if self.trade_direction == OrderDirection.BUY:
            delta -= self.trade_price * self.trade_volume
            delta -= self.trade_cost
        elif self.trade_direction == OrderDirection.SELL:
            delta += self.trade_price * self.trade_volume
            delta -= self.trade_cost
        return two_decimal_price(delta)

    def __repr__(self):
        return f"< volume: {self.trade_volume} price: {self.trade_price} cost: {self.trade_cost} >"


class OrderActionScope:
    """Action scope of the Order."""

    def __init__(self, min_buy_volume, max_buy_volume, min_sell_volume, max_sell_volume, supported_order_mode):
        self.min_buy_volume = min_buy_volume
        self.max_buy_volume = max_buy_volume
        self.min_sell_volume = min_sell_volume
        self.max_sell_volume = max_sell_volume
        self.supported_order_mode = supported_order_mode

    def __repr__(self):
        return (
            f"buy scope: {self.min_buy_volume}~{self.max_buy_volume}"
            + f"sell scope: {self.min_sell_volume}~{self.max_sell_volume} order types: {self.supported_order_mode}"
        )


class CancelActionScope:
    """Action scope of a cancel order action."""

    def __init__(self, available_orders):
        self.available_orders = available_orders

    def __repr__(self):
        return f"available orders: {self.available_orders}"


class DecisionEvent:
    """Decision event given by environment, agent make an action according to the decision event.

    Args:
        tick (int): current tick of decision.
        action_scope (any): scope of the action
    """

    def __init__(self, tick: int, action_scope: any):

        self.tick = tick
        self.action_scope = action_scope

    def __repr__(self):
        return f"DecisionEvent tick: {self.tick}, action scope: {self.action_scope}"


class OrderDecisionEvent(DecisionEvent):
    """Order decision event given to agent, contains the info for making an order action.

    Args:
        tick (int): current tick of decision.
        stock_index (int): stock index to make the order.
        action_scope (OrderActionScope): scope of the order.
    """

    def __init__(self, tick: int, stock_index: int, action_scope: OrderActionScope):
        super().__init__(tick, action_scope)
        self.stock_index = stock_index

    def __repr__(self):
        return (
            f"<Order decision envet tick: {self.tick}, stock index: {self.stock_index},"
            + f"action scope: {self.action_scope}>"
        )


class CancelDecisionEvent(DecisionEvent):
    """Cancel decision event given to agent, contains the actions index can be canceled.

    Args:
        tick (int): current tick of decision.
        action_scope (CancelActionScope): scope of the order.
    """

    def __init__(self, tick: int, action_scope: CancelActionScope):
        super().__init__(tick, action_scope)


class Action(ABC):
    """The action is the reply to the decison event, tells the environment what to do in the next setp.

    Args:
        tick (int): The tick when the action is created.
        id (int): (optional)The id of the action, if not specified, it will be automatically generated.
        life_time (int): (optional)The life time of the action, if not specified, it will be set to 1.
    """
    idx = 0

    def __init__(
        self,
        tick: int = 0,
        id: int = None, life_time: int = 1
    ):

        self.decision_tick = tick
        self.finish_tick = None
        self.state = ActionState.REQUEST
        if id is not None:
            self.id = id
        else:
            self.id = Action.idx
            Action.idx += 1
        self.remaining_life_time = life_time
        self.action_result = None
        self.comment = ""

    def finish(self, tick: int, state: ActionState = ActionState.FINISH, result: any = None, comment: str = None):
        self.finish_tick = tick
        self.state = state
        self.action_result = result
        self.comment = comment
        self.remaining_life_time = 0

    def __repr__(self):
        return (
            f"< Action ID:{self.id} start: {self.decision_tick} finished: {self.finish_tick} state: {self.state}"
            + f"remaining life time:{self.remaining_life_time} comment: {self.comment}>"
        )


class Order(Action):
    """The order is an action tells the environment to buy or sell the assets.

    Args:
        stock_index (int): The index of the stock to buy or sell.
        order_volume (int): The volume of the stock to buy or sell.
        order_mode (OrderMode): The mode of the order.
        order_direction (OrderDirection): The direction of the order.
        tick (int): The tick when the order is created.
        life_time (int): (optional)The life time of the order, if not specified, it will be set to 1.
    """

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
        return (
            f"{super().__repr__()}\n< Order item: {self.stock_index} volume: {self.order_volume}"
            + f"direction: {self.order_direction} >"
        )


class MarketOrder(Order):
    """The market order is an order tells the environment to buy or sell the assets with a market order.

    Args:
        tick (int): The tick when the order is created.
        stock_index (int): The index of the stock to buy or sell.
        order_volume (int): The volume of the stock to buy or sell.
        order_direction (OrderDirection): The direction of the order.
        life_time (int): (optional)The life time of the order, if not specified, it will be set to 1.
    """

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
        return triggered


class LimitOrder(Order):
    """The limit order is an order tells the environment to buy or sell the assets with a limit order.

    Args:
        tick (int): The tick when the order is created.
        stock_index (int): The index of the stock to buy or sell.
        order_volume (int): The volume of the stock to buy or sell.
        order_direction (OrderDirection): The direction of the order.
        limit (float): The limit price of the order.
        life_time (int): (optional)The life time of the order, if not specified, it will be set to 1.
    """

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
        return triggered


class StopOrder(Order):
    """The stop order is an order tells the environment to buy or sell the assets with a stop order.

    Args:
        tick (int): The tick when the order is created.
        stock_index (int): The index of the stock to buy or sell.
        order_volume (int): The volume of the stock to buy or sell.
        order_direction (OrderDirection): The direction of the order.
        stop (float): The stop price of the order.
        life_time (int): (optional)The life time of the order, if not specified, it will be set to 1.
    """

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
        return triggered


class StopLimitOrder(Order):
    """The stop limit order is an order tells the environment to buy or sell the assets with a stop limit order.

    Args:
        tick (int): The tick when the order is created.
        stock_index (int): The index of the stock to buy or sell.
        order_volume (int): The volume of the stock to buy or sell.
        order_direction (OrderDirection): The direction of the order.
        stop (float): The stop price of the order.
        limit (float): The limit price of the order.
        life_time (int): (optional)The life time of the order, if not specified, it will be set to 1.
    """

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

        return triggered


class Cancel(Action):
    """Cancel order is an action specifies the order to cancel."""

    def __init__(self, order: Order, tick: int):
        super().__init__(
            tick=tick, life_time=1
        )
        self.order = order


def two_decimal_price(input_price: float) -> float:
    """Keep 2 decimal places of a float number."""
    return int(input_price * 100) / 100


def get_cn_stock_data_tick(start_date: str) -> int:
    """Convert the start date of an experiment to the tick of the finance scenario."""
    ret = None
    tzone = "Asia/Shanghai"
    default_start_dt = "1991-01-01"
    default_time_unit = "d"
    is_dt, dt = is_datetime(start_date, tzone)
    if is_dt:
        _, start_dt = is_datetime(default_start_dt, tzone)
        dt_seconds = calendar.timegm(dt.timetuple())
        start_dt_seconds = calendar.timegm(start_dt.timetuple())
        delta_seconds = dt_seconds - start_dt_seconds
        seconds_per_unit = unit_seconds(default_time_unit)
        ret = int((delta_seconds) / seconds_per_unit)
    return ret


def _get_stock_start_timestamp(start_date: str = "1991-01-01", tzone: str = "Asia/Shanghai") -> int:
    ret = None
    default_start_dt = "1970-01-01"
    default_start_tzone = "UTC"
    default_time_unit = "s"
    is_dt, dt = is_datetime(start_date, tzone)
    if is_dt:
        _, start_dt = is_datetime(default_start_dt, default_start_tzone)
        dt_seconds = calendar.timegm(dt.timetuple())
        start_dt_seconds = calendar.timegm(start_dt.timetuple())
        delta_seconds = dt_seconds - start_dt_seconds
        seconds_per_unit = unit_seconds(default_time_unit)
        ret = int(delta_seconds / seconds_per_unit)
    return ret
