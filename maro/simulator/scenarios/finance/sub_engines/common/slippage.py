from enum import Enum
import math

from maro.simulator.scenarios.finance.sub_engines.common.trader import TradeConstrain
from maro.simulator.scenarios.finance.common import OrderMode, Action


class SlippageType(Enum):
    # TODO: zhanyu add correct slippage
    no_slippage = 0
    by_money_slippage = 1
    by_volume_slippage = 2
    by_trade_slippage = 3


class Slippage():
    __slippage_type = None

    def __init__(self):
        pass

    def execute(self, order_action: Action, cur_data: dict):
        pass

    @property
    def slippage_type(self) -> SlippageType:
        return self.__slippage_type


class ByMoneySlippage(Slippage):
    __slippage_rate = 0

    def __init__(self, slippage_rate: float = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_money_slippage
        self.__slippage_rate = slippage_rate

    def execute(self, order_action: Action, cur_data: dict) -> float:
        order_direction = 1
        if order_action.number < 0:
            order_direction = -1
        actual_price = round(cur_data[order_action.item_index].opening_price*(1+self.__slippage_rate*order_direction/2), 2)
        return actual_price


class ByVolumeSlippage(Slippage):
    __pre_volume_fee = 0

    def __init__(self, pre_volume_fee: float = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_volume_slippage
        self.__pre_volume_fee = pre_volume_fee

    def execute(self, order_action: Action, cur_data: dict) -> float:
        return round(abs(order_action.volume)*self.__pre_volume_fee, 2)


class ByTradeSlippage(Slippage):
    __pre_trade_fee = 0

    def __init__(self, pre_trade_fee: float = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_volume_slippage
        self.__pre_trade_fee = pre_trade_fee

    def execute(self, order_action: Action, cur_data: dict) -> float:
        return round(self.__pre_trade_fee, 2)
