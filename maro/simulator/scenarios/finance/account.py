"""Used to maintain stock/futures, one account per episode"""

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

    def take_trade(self, trade_result: TradeResult):
        self._remaining_money -= trade_result.total_cost

    def calc_reward(self):
        # TODO: zhanyu to fill the logic
        # - last tick
        pass

    def reset(self):
        self._remaining_money = self._money
        self._last_total_money = self._money
