"""Used to maintain stock/futures, one account per episode"""
from collections import OrderedDict

from maro.backends.frame import SnapshotList
from maro.backends.frame import node, NodeBase, NodeAttribute
from maro.simulator.scenarios.finance.common import Action, ActionState

from .common import TradeResult
from .currency import CurrencyType, Exchanger


class AssetsAccessor:
    """This wrapper used to provide interfaces to access asset related quering inside one object"""

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
    def __init__(self, account_snapshot: SnapshotList, sub_engines_snapshots: dict):
        self._account_snapshots = account_snapshot
        self._assets_accessor = AssetsAccessor(sub_engines_snapshots)

    def __getattribute__(self, name):
        __dict__ = object.__getattribute__(self, "__dict__")

        if name == "assets":
            return __dict__["_assets_accessor"]

        return __dict__["_account_snapshots"].__getattribute__(name)


@node("account")
class Account(NodeBase):
    remaining_money = NodeAttribute("f")

    def __init__(self):
        self.action_history = OrderedDict()  # TODO: later
        self._last_total_money = 0

    def set_init_state(self, sub_account_dict: dict, money: float, currency: str = 'CNY'):
        # NOTE: the sub_account_dict, the key is the sub-engine name, value is SubAccount instance
        self._sub_account_dict = sub_account_dict
        self._currency = CurrencyType[currency]
        self._money = money
        self.remaining_money = self._money

    def take_trade(self, trade_result: TradeResult, cur_data: list, cur_engine: str):
        self._last_total_money = self.total_money
        if trade_result.is_trade_accept and trade_result.is_trade_trigger:
            self._sub_account_dict[cur_engine].take_trade(trade_result, cur_data)

    def calc_reward(self):
        # TODO: zhanyu to update the logic
        # - last tick
        reward = self.total_money - self._last_total_money
        print("reward:", reward)
        return reward

    def reset(self):
        # self._last_total_money = self._money * self._leverage
        # self._sub_account = OrderedDict()
        self.action_history.clear()
        self.remaining_money = self._money

        for _, sub_account in self._sub_account_dict.items():
            sub_account.reset()

    @property
    def total_money(self):
        return sum([x.total_money for x in self._sub_account_dict.values()])

    @property
    def currency(self):
        return self._currency

    def transfer(self, target: str, amount: float, tick: int, enable_leverage=False):
        """
        if amount > 0, transfer from main account to sub-account, currency use main account
        if amount < 0, transfer from sub-account to main account, currency use sub-account
        """
        result = 0
        if target in self._sub_account_dict:
            target_account: SubAccount = self._sub_account_dict[target]
            is_from_main_account = True if amount > 0 else False
            if is_from_main_account:
                if self.remaining_money < amount:
                    amount = self.remaining_money
                exchanger: Exchanger = Exchanger.get_exchanger()
                exchanged = exchanger.exchange_from_by_tick(self.currency, target_account.currency, amount, tick)
                result = exchanged
                self.remaining_money -= amount
                target_account.transfer(result, enable_leverage)
                print("transfer from main account to sub-account", amount, result)
            else:
                amount = abs(amount)
                amount = target_account.transfer(-amount)
                exchanger: Exchanger = Exchanger.get_exchanger()
                exchanged = exchanger.exchange_from_by_tick(target_account.currency, self.currency, amount, tick)
                result = exchanged
                self.remaining_money += result
                print("transfer from sub-account to main account", amount, result)

        return amount, result

    def take_action(self, action: Action, tick: int):
        amount, result = self.transfer(action.sub_engine_name, action.number, tick)
        if result != 0:
            action.state = ActionState.success
        else:
            action.state = ActionState.failed
        action.finish_tick = tick
        action.action_result = (amount, result)


@node("sub_account")
class SubAccount(NodeBase):
    """Used to maintain money for different market (sub-engine)"""
    remaining_money = NodeAttribute("f")
    total_money = NodeAttribute("f")
    _currency = None
    _leverage = 1
    _min_leverage_rate = 0
    _init_self_money = 0
    _init_loan_money = 0
    _self_money = 0
    _loan_money = 0

    def __init__(self):
        pass

    def set_init_state(
            self, init_money: float, currency: str = 'CNY',
            leverage: float = 1, min_leverage_rate: float = 0
    ):
        self._leverage = leverage
        self._min_leverage_rate = min_leverage_rate
        self._currency = CurrencyType[currency]
        self._init_self_money = init_money
        self._self_money = self._init_self_money
        self._init_loan_money = self._init_self_money * (self._leverage - 1)
        self._loan_money = self._init_loan_money

        self.total_money = self._self_money + self._loan_money
        self.remaining_money = self.total_money

    def reset(self):
        self._self_money = self._init_self_money
        self._init_loan_money = self._init_self_money * (self._leverage - 1)
        self._loan_money = self._init_loan_money
        self.total_money = self._self_money + self._loan_money
        self.remaining_money = self.total_money

    def take_trade(self, trade_result: TradeResult, cur_data: list):
        cur_position = 0
        for stock in cur_data:
            cur_position += stock.closing_price * stock.account_hold_num
        self.remaining_money -= trade_result.total_cost
        self.total_money = self.remaining_money + cur_position

    @property
    def currency(self):
        return self._currency

    def leverage_alert(self):
        ret = False
        if self._loan_money > 0 and self.total_money / (self._loan_money) < self._min_leverage_rate:
            ret = True
        return ret

    def transfer(self, amount: float, enable_leverage: bool = False):
        result = 0
        if amount > 0:
            result = amount
            if not enable_leverage:
                self._self_money += result
                self.remaining_money += result
                self.total_money += result
            else:
                if not self.leverage_alert():
                    self._self_money += result
                    self._loan_money += result * self._leverage
                    self.remaining_money += result * self._leverage
                    self.total_money += result * self._leverage
                else:
                    print("warning: failed to leverage due to leverage alert")
                    self._self_money += result
                    self.remaining_money += result
                    self.total_money += result
        if amount < 0:
            amount = abs(amount)
            if not self.leverage_alert():
                available_amount = min(
                    [self.total_money - self._loan_money * self._min_leverage_rate, self.remaining_money]
                )
                if available_amount > 0:
                    if amount > available_amount:
                        result = available_amount
                    else:
                        result = amount
                    self.remaining_money -= result
                    self.total_money -= result
                else:
                    print("warning: no available money to transfer out")
            else:
                print("warning: failed to transfer out due to leverage alert")
        return result
