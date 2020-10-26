"""Used to maintain stock/futures, one account per episode"""
from collections import OrderedDict
from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.simulator.scenarios.finance.common.common import TradeResult


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

    def take_trade(self, trade_result: TradeResult, cur_data: list):
        self._last_total_money = self.total_money
        if trade_result.is_trade_accept and trade_result.is_trade_trigger:
            cur_position = 0
            for stock in cur_data:
                cur_position += stock.closing_price * stock.account_hold_num
            self.remaining_money -= trade_result.total_cost
            self.total_money = self.remaining_money + cur_position

    def calc_reward(self):
        reward = self.total_money - self._last_total_money
        print("reward:", reward)
        return reward

    def reset(self):
        self._last_total_money = 0
        self.remaining_money = self._money
        self.total_money = self._money
        self.action_history.clear()
