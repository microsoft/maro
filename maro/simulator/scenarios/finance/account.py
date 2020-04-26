"""Used to maintain stock/futures, one account per episode"""
from collections import OrderedDict

import numpy as np

from maro.simulator.frame import Frame, FrameNodeType, SnapshotList
from maro.simulator.scenarios.entity_base import (EntityBase, FloatAttribute,
                                                  FrameBuilder, IntAttribute,
                                                  frame_node)
from maro.simulator.scenarios.finance.common import Action, FinanceType

from .common import TradeResult


# snapshots.account.assets[tick: "sub-engine"]
class AssetsAccessor:
    def __init__(self, sub_engines_snapshots: dict):
        self._sub_engine_snapshots = sub_engines_snapshots

    def __getitem__(self, key: slice):
        ticks = key.start
        engine_names = key.stop

        if ticks is None:
            raise "ticks cannot be None for asset querying"

        if type(ticks) is not list:
            ticks = [ticks]

        if engine_names is None:
            engine_names = [k for k in self._sub_engine_snapshots.keys()]

        if type(engine_names) is not list:
            engine_names = [engine_names]

        results = {}

        for sub_name in engine_names:
            sub_snapshot: SnapshotList = self._sub_engine_snapshots[sub_name]

            results[sub_name] = sub_snapshot.static_nodes[ticks::"account_hold_num"]

        return results

class AccountSnapshotWrapper:
    def __init__(self, account_snapshots: SnapshotList, sub_engines_snapshots: dict):
        self._account_snapshots = account_snapshots
        self._assets_accessor = AssetsAccessor(sub_engines_snapshots)

    def __getattribute__(self, name):
        __dict__ = object.__getattribute__(self, "__dict__")        

        if name == "assets":
            return __dict__["_assets_accessor"]

        return __dict__["_account_snapshots"].__getattribute__(name)

@frame_node(FrameNodeType.STATIC)
class Account(EntityBase):
    remaining_money = FloatAttribute()
    total_money = FloatAttribute()

    # all stock
    # all future
    def __init__(self, snapshots, frame: Frame, money: float):
        super().__init__(frame, 0)
        # NOTE: the snapshots is wrapper of snapshots of sub-engines,
        # you can access them by sub-engine name like: snapshots.china to calculate reward
        self._money = money
        self._trade_history = []  # TODO: later
        self.remaining_money = money
        self._last_total_money = money
        self.total_money = money
        self._sub_account = OrderedDict()

    def take_trade(self, trade_result: TradeResult, cur_data: list, cur_engine: str):
        self._last_total_money = self.total_money
        self.remaining_money -= trade_result.total_cost
        self.total_money  = self.remaining_money
        self._sub_account[cur_engine] = 0
        for stock in cur_data:
            self._sub_account[cur_engine] += stock.closing_price * stock.account_hold_num
        for engine in self._sub_account.keys():
            self.total_money += self._sub_account[engine]


    def calc_reward(self):
        # TODO: zhanyu to update the logic
        # - last tick
        reward = self.total_money - self._last_total_money
        print("reward:", reward)
        return reward

    def reset(self):
        self.remaining_money = self._money
        self._last_total_money = self._money
        self.total_money = self._money
        self._sub_account = OrderedDict()
