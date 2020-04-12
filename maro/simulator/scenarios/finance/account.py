"""Used to maintain stock/futures, one account per episode"""

from maro.simulator.frame import SnapshotList
from maro.simulator.scenarios.finance.common import Action, FinanceType

class Account:
    def __init__(self, money: float, snapshot_list: SnapshotList):
        self._stock_number = 0
        self._futures_number = 0
        self._monty = money
        self._snapshot_list = snapshot_list

    def take_action(self, action: Action):

        # NOTE: we ignore the money cost for now

        if action.type == FinanceType.stock:
            self._stock_number += action.number

    def calc_reward(self):
        pass

class AcountList:
    def __init__(self, init_money: float, snapshot_list: SnapshotList):
        self._accounts=[]
        self._init_money = init_money
        self._snapshot_list = snapshot_list

    def new_account(self):
        return Account(self._snapshot_list)