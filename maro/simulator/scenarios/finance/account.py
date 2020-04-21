"""Used to maintain stock/futures, one account per episode"""
from collections import OrderedDict

from .common import TradeResult
from maro.simulator.frame import SnapshotList, Frame, FrameNodeType
from maro.simulator.scenarios.finance.common import Action, FinanceType
from maro.simulator.scenarios.entity_base import EntityBase, IntAttribute, FloatAttribute, FrameBuilder, frame_node


@frame_node(FrameNodeType.STATIC)
class Account(EntityBase):
    remaining_money = FloatAttribute()

    # all stock
    # all future

    def __init__(self, snapshots, money: float):
        # NOTE: the snapshots is wrapper of snapshots of sub-engines,
        # you can access them by sub-engine name like: snapshots.china to calculate reward
        self._money = money
        self._trade_history = []  # TODO: later
        self._remaining_money = money
        self._last_total_money = money
        self._total_money = money
        self._sub_account = OrderedDict()

    def take_trade(self, trade_result: TradeResult, cur_data: list, cur_engine: str):
        self._last_total_money = self._total_money
        self._remaining_money -= trade_result.total_cost
        self._total_money  = self._remaining_money
        self._sub_account[cur_engine] = 0
        for stock in cur_data:
            self._sub_account[cur_engine] += stock.closing_price * stock.account_hold_num
        for engine in self._sub_account.keys():
            self._total_money += self._sub_account[engine]


    def calc_reward(self):
        # TODO: zhanyu to update the logic
        # - last tick
        reward = self._total_money - self._last_total_money
        print("reward:", reward)
        return reward

    def reset(self):
        self._remaining_money = self._money
        self._last_total_money = self._money
        self._total_money = self._money
        self._sub_account = OrderedDict()
