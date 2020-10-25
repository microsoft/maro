from enum import Enum

from maro.simulator.scenarios.finance.common.common import Action


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

    def execute(self, order_action: Action, cur_data: dict, deal_price: float):
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

    def execute(self, order_action: Action, cur_data: dict, deal_price: float) -> float:
        order_direction = 1
        if order_action.number < 0:
            order_direction = -1
        actual_price = deal_price * (1 + self.__slippage_rate * order_direction / 2)
        return actual_price


class ByVolumeSlippage(Slippage):
    __pre_volume_fee = 0

    def __init__(self, pre_volume_fee: float = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_volume_slippage
        self.__pre_volume_fee = pre_volume_fee

    def execute(self, order_action: Action, cur_data: dict, deal_price: float) -> float:
        return round(abs(order_action.volume) * self.__pre_volume_fee, 2) # problem


class ByTradeSlippage(Slippage):
    __pre_trade_fee = 0

    def __init__(self, pre_trade_fee: float = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_trade_slippage
        self.__pre_trade_fee = pre_trade_fee

    def execute(self, order_action: Action, cur_data: dict, deal_price: float) -> float:
        return round(self.__pre_trade_fee, 2) # problem
