"""Used to maintain stock/futures"""
from collections import OrderedDict

from maro.backends.frame import NodeAttribute, NodeBase, node
from maro.simulator.scenarios.finance.common.common import Order, OrderDirection, TradeResult, two_decimal_price


@node("account")
class Account(NodeBase):
    """Account node definition in frame."""
    remaining_cash = NodeAttribute("f")
    total_assets_value = NodeAttribute("f")

    def __init__(self):
        self.action_history = OrderedDict()

    def set_init_state(self, init_money: float):
        """Set initialize state, that will be used after frame reset.

        Args:
            init_money (float): Initial money in the account.
        """
        self._init_money = init_money
        self.remaining_cash = self._init_money
        self.total_assets_value = self._init_money

    def take_trade(self, order: Order, trade_result: TradeResult):
        if trade_result:
            if order.direction == OrderDirection.buy:
                self.remaining_cash -= \
                    two_decimal_price(trade_result.trade_number * trade_result.price_per_item + \
                    trade_result.commission + trade_result.tax)
            else:
                self.remaining_cash += \
                    two_decimal_price(trade_result.trade_number * trade_result.price_per_item - \
                    trade_result.commission - trade_result.tax)

    def update_assets_value(self, cur_data: list):
        cur_position = 0
        for stock in cur_data:
            cur_position += stock.last_closeing * stock.account_hold_num
        self.total_assets_value = two_decimal_price(self.remaining_cash + cur_position)

    def reset(self):
        self.remaining_cash = self._init_money
        self.total_assets_value = self._init_money
        self.action_history.clear()
