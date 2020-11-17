from collections import OrderedDict
from typing import List

from maro.backends.frame import NodeAttribute, NodeBase, node

from .common import TradeResult, two_decimal_price
from .stock import Stock


@node("account")
class Account(NodeBase):
    """Account node definition in frame, used to maintain cash changing."""
    remaining_cash = NodeAttribute("f")
    net_assets_value = NodeAttribute("f")

    def set_init_state(self, init_cash: float):
        """Set initial state, that will be used after frame reset.

        Args:
            init_cash (float): Initial cash in the account.
        """
        self._init_cash = init_cash
        self.remaining_cash = self._init_cash
        self.net_assets_value = self._init_cash

    def take_trade(self, trade_result: TradeResult):
        if trade_result:
            self.remaining_cash += two_decimal_price(trade_result.cash_delta)

    def update_assets_value(self, stocks: List[Stock]):
        assets_value = 0
        for stock in stocks:
            assets_value += stock.closing_price * stock.account_hold_num
        self.net_assets_value = two_decimal_price(self.remaining_cash + assets_value)

    def reset(self):
        self.remaining_cash = self._init_cash
        self.net_assets_value = self._init_cash
