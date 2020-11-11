"""Used to maintain stock/futures"""
from collections import OrderedDict

from maro.backends.frame import NodeAttribute, NodeBase, node
from maro.simulator.scenarios.finance.common.common import Order, TradeResult, two_decimal_price


@node("account")
class Account(NodeBase):
    """Account node definition in frame."""
    remaining_cash = NodeAttribute("f")
    net_assets_value = NodeAttribute("f")

    def __init__(self):
        self.action_history = OrderedDict()

    def set_init_state(self, init_money: float):
        """Set initialize state, that will be used after frame reset.

        Args:
            init_money (float): Initial money in the account.
        """
        self._init_money = init_money
        self.remaining_cash = self._init_money
        self.net_assets_value = self._init_money

    def take_trade(self, order: Order, trade_result: TradeResult):
        if trade_result:
            self.remaining_cash += two_decimal_price(trade_result.cash_delta)

    def update_assets_value(self, cur_data: list):
        assets_value = 0
        for stock in cur_data:
            assets_value += stock.last_closeing * stock.account_hold_num
        self.net_assets_value = two_decimal_price(self.remaining_cash + assets_value)

    def reset(self):
        self.remaining_cash = self._init_money
        self.net_assets_value = self._init_money
        self.action_history.clear()
