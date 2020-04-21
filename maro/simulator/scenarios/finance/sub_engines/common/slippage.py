from enum import Enum

class SlippageType(Enum):
    no_slippage = 0
    by_money_slippage = 1
    by_volume_slippage = 2
    by_trade_slippage = 3

class Slippage():
    __slippage_type = None
    def __init__(self):
        pass

    def execute(self, order_action, curr_data):
        pass

    @property
    def slippage_type(self):
        return self.__slippage_type

class ByMoneySlippage(Slippage):
    __slippage_rate = 0

    def __init__(self, slippage_rate = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_money_slippage
        self.__slippage_rate = slippage_rate

    def execute(self, order_action, curr_data):
        return curr_data[order_action.asset].closing_price*(1+self.__slippage_rate)

class ByVolumeSlippage(Slippage):
    __pre_volume_fee = 0

    def __init__(self, pre_volume_fee = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_volume_slippage
        self.__pre_volume_fee = pre_volume_fee

    def execute(self, order_action, curr_data):
        return abs(order_action.volume)*self.__pre_volume_fee

class ByTradeSlippage(Slippage):
    __pre_trade_fee = 0

    def __init__(self, pre_trade_fee = 0):
        Slippage.__init__(self)
        self.__slippage_type = SlippageType.by_volume_slippage
        self.__pre_trade_fee = pre_trade_fee

    def execute(self, order_action, curr_data):
        return self.__pre_trade_fee


        

    

