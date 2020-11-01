from enum import Enum

from maro.simulator.scenarios.finance.common.common import Action, OrderDirection


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

    def execute(self, direction: OrderDirection, volume: int, base_price: float, market_volume: int):
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

    def execute(self, direction: OrderDirection, volume: int, base_price: float, market_volume: int) -> float:
        delta_price = base_price * self.__slippage_rate / 2
        if direction == OrderDirection.buy:
            actual_price = base_price + delta_price
        else:
            actual_price = base_price - delta_price
        return actual_price
