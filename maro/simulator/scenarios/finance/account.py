"""Used to maintain stock/futures, one account per episode"""
from collections import OrderedDict

from maro.backends.frame import NodeAttribute, NodeBase, node
from maro.simulator.scenarios.finance.common.common import Order, OrderDirection, TradeResult


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

    def take_trade(self, order: Order, trade_result: TradeResult, cur_data: list):
        if trade_result:
            if order.direction == OrderDirection.buy:
                self.remaining_money -= trade_result.trade_number * trade_result.price_per_item + trade_result.tax
            else:
                self.remaining_money += trade_result.trade_number * trade_result.price_per_item - trade_result.tax

    def update_position(self, cur_data: list):
        cur_position = 0
        for stock in cur_data:
            cur_position += stock.last_closeing * stock.account_hold_num
        self.total_money = self.remaining_money + cur_position

    def reset(self):
        self.remaining_money = self._money
        self.total_money = self._money
        self.action_history.clear()
