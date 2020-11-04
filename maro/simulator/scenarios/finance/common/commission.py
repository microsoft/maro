from enum import Enum

from maro.simulator.scenarios.finance.common.common import OrderDirection


class CommissionType(Enum):
    no_commission = 0
    stamp_tax_commission = 1
    by_money_commission = 2
    by_volume_commission = 3
    by_trade_commission = 4


class Commission():
    _commission_type = None
    _min_fee = 0

    def __init__(self, min_fee):
        self.min_fee = min_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int):
        pass

    @property
    def commission_type(self):
        return self._commission_type

    @commission_type.setter
    def commission_type(self, value):
        self._commission_type = value

    @property
    def min_fee(self):
        return self._min_fee

    @min_fee.setter
    def min_fee(self, value):
        self._min_fee = value


class ByMoneyCommission(Commission):
    _fee_rate = 0

    def __init__(self, fee_rate: float = 0, min_fee: float = 0):
        Commission.__init__(self, min_fee)
        self.commission_type = CommissionType.by_money_commission
        self.fee_rate = fee_rate
        self.min_fee = min_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        return round(max(actual_price * actual_volume * self.fee_rate, self.min_fee), 2)

    @property
    def fee_rate(self):
        return self._fee_rate

    @fee_rate.setter
    def fee_rate(self, value):
        self._fee_rate = value


class ByVolumeCommission(Commission):
    _pre_volume_fee = 0

    def __init__(self, pre_volume_fee: float = 0, min_fee: float = 0):
        Commission.__init__(self, min_fee)
        self.commission_type = CommissionType.by_volume_commission
        self.pre_volume_fee = pre_volume_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        return round(max(actual_volume * self.pre_volume_fee, self.min_fee), 2)

    @property
    def pre_volume_fee(self):
        return self._pre_volume_fee

    @pre_volume_fee.setter
    def pre_volume_fee(self, value):
        self._pre_volume_fee = value


class ByTradeCommission(Commission):
    _pre_trade_fee = 0

    def __init__(self, pre_trade_fee: float = 0, min_fee: float = 0):
        Commission.__init__(self, min_fee)
        self.commission_type = CommissionType.by_volume_commission
        self.pre_trade_fee = pre_trade_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        return round(max(self.pre_trade_fee, self.min_fee), 2)

    @property
    def pre_trade_fee(self):
        return self._pre_trade_fee

    @pre_trade_fee.setter
    def pre_trade_fee(self, value):
        self._pre_trade_fee = value


class StampTaxCommission(ByMoneyCommission):

    def __init__(self, tax_rate: float = 0, min_fee: float = 0):
        ByMoneyCommission.__init__(self, tax_rate, min_fee)
        self.commission_type = CommissionType.stamp_tax_commission

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        if direction == OrderDirection.sell:
            return round(max(actual_price * actual_volume * self.tax_rate, self.min_fee), 2)
        else:
            return 0

    @property
    def tax_rate(self):
        return self.fee_rate

    @tax_rate.setter
    def tax_rate(self, value):
        self.fee_rate = value
