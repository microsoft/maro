"""Used to maintain stock/futures, one account per episode"""
from collections import OrderedDict
from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.simulator.scenarios.finance.common.common import TradeResult, Order, OrderDirection


@node("account")
class Account(NodeBase):
    remaining_money = NodeAttribute("f")
    total_money = NodeAttribute("f")

    def __init__(self):
        self.action_history = OrderedDict()

    def set_init_state(self, init_money: float):
        self._money = init_money
        self.remaining_money = self._money
        self.total_money = self._money
        self._last_total_money = self._money

    def take_trade(self, order: Order, trade_result: TradeResult, cur_data: list):
        self._last_total_money = self.total_money
        if trade_result:
            cur_position = 0
            for stock in cur_data:
                cur_position += stock.closing_price * stock.account_hold_num
            if order.direction == OrderDirection.buy:
                self.remaining_money -= trade_result.trade_number * trade_result.price_per_item + trade_result.tax
            else:
                self.remaining_money += trade_result.trade_number * trade_result.price_per_item - trade_result.tax
            self.total_money = self.remaining_money + cur_position

    def reset(self):
        self._last_total_money = 0
        self.remaining_money = self._money
        self.total_money = self._money
        self.action_history.clear()
