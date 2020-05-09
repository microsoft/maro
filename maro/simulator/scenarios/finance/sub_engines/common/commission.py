from enum import Enum


class CommissionType(Enum):
    no_commission = 0
    stamp_tax_commission = 1
    by_money_commission = 2
    by_volume_commission = 3
    by_trade_commission = 4


class Commission():
    __commission_type = None
    __min_fee = 0

    def __init__(self, min_fee):
        self.__min_fee = min_fee

    def execute(self, actual_price: float, actual_volume: int):
        pass

    @property
    def commission_type(self):
        return self.__commission_type

    @property
    def min_fee(self):
        return self.__min_fee


class ByMoneyCommission(Commission):
    __fee_rate = 0

    def __init__(self, fee_rate: float = 0, min_fee: float = 0):
        Commission.__init__(self, min_fee)
        self.__commission_type = CommissionType.by_money_commission
        self.__fee_rate = fee_rate
        self.__min_fee = min_fee

    def execute(self, actual_price: float, actual_volume: int)-> float:
        return round(max(actual_price*abs(actual_volume)*self.__fee_rate, self.__min_fee), 2)

    @property
    def fee_rate(self):
        return self.__fee_rate


class ByVolumeCommission(Commission):
    __pre_volume_fee = 0

    def __init__(self, pre_volume_fee: float = 0, min_fee: float = 0):
        Commission.__init__(self, min_fee)
        self.__commission_type = CommissionType.by_volume_commission
        self.__pre_volume_fee = pre_volume_fee

    def execute(self, actual_price: float, actual_volume: int)-> float:
        return round(max(abs(actual_volume)*self.__pre_volume_fee, self.__min_fee), 2)

    @property
    def pre_volume_fee(self):
        return self.__pre_volume_fee


class ByTradeCommission(Commission):
    __pre_trade_fee = 0

    def __init__(self, pre_trade_fee: float = 0, min_fee: float = 0):
        Commission.__init__(self, min_fee)
        self.__commission_type = CommissionType.by_volume_commission
        self.__pre_trade_fee = pre_trade_fee

    def execute(self, actual_price: float, actual_volume: int)-> float:
        return round(max(self.__pre_trade_fee, self.__min_fee), 2)

    @property
    def pre_trade_fee(self):
        return self.__pre_trade_fee


class StampTaxCommission(ByMoneyCommission):

    def __init__(self, tax_rate: float = 0, min_fee: float = 0):
        ByMoneyCommission.__init__(self, tax_rate, min_fee)
        self.__commission_type = CommissionType.stamp_tax_commission

    def execute(self, actual_price: float, actual_volume: int)-> float:
        if actual_volume < 0:
            return round(max(actual_price*abs(actual_volume)*self.tax_rate, self.min_fee), 2)
        else:
            return 0

    @property
    def tax_rate(self):
        return self.fee_rate
