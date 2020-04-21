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

    def execute(self, actual_price, actual_volume):
        pass

    @property
    def commission_type(self):
        return self.__commission_type

class ByMoneyCommission(Commission):
    __fee_rate = 0
    __min_fee = 0
    def __init__(self, fee_rate = 0, min_fee = 0):
        Commission.__init__(self, min_fee)
        self.__commission_type = CommissionType.by_money_commission
        self.__fee_rate = fee_rate
        self.__min_fee = min_fee

    def execute(self, actual_price, actual_volume):
        return max(actual_price*abs(actual_volume)*self.__fee_rate, self.__min_fee)

class ByVolumeCommission(Commission):
    __pre_volume_fee = 0

    def __init__(self, pre_volume_fee = 0, min_fee = 0):
        Commission.__init__(self, min_fee)
        self.__commission_type = CommissionType.by_volume_commission
        self.__pre_volume_fee = pre_volume_fee

    def execute(self, actual_price, actual_volume):
        return max(abs(actual_volume)*self.__pre_volume_fee, self.__min_fee)

class ByTradeCommission(Commission):
    __pre_trade_fee = 0

    def __init__(self, pre_trade_fee = 0, min_fee = 0):
        Commission.__init__(self, min_fee)
        self.__commission_type = CommissionType.by_volume_commission
        self.__pre_trade_fee = pre_trade_fee

    def execute(self, actual_price, actual_volume):
        return max(self.__pre_trade_fee, self.__min_fee)

class StampTaxCommission(ByMoneyCommission):

    def __init__(self, tax_rate = 0, min_fee = 0):
        ByMoneyCommission.__init__(self, tax_rate, min_fee)
        self.__commission_type = CommissionType.stamp_tax_commission

    def execute(self, actual_price, actual_volume):
        if actual_volume < 0:
            return max(actual_price*abs(actual_volume)*self.__fee_rate, self.__min_fee)
        else:
            return  0
        

    

