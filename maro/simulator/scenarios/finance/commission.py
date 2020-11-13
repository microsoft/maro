from maro.simulator.scenarios.finance.common import OrderDirection, two_decimal_price


class Commission():
    _min_fee = 0

    def __init__(self, min_fee):
        self._min_fee = min_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int):
        pass

    @property
    def min_fee(self):
        return self._min_fee


class FixedCommission(Commission):
    _fee_rate = 0

    def __init__(self, fee_rate: float = 0, min_fee: float = 0):
        super().__init__(min_fee)
        self._fee_rate = fee_rate
        self._min_fee = min_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        return two_decimal_price(max(actual_price * actual_volume * self.fee_rate, self.min_fee))

    @property
    def fee_rate(self):
        return self._fee_rate


class ByVolumeCommission(Commission):
    _pre_volume_fee = 0

    def __init__(self, pre_volume_fee: float = 0, min_fee: float = 0):
        super().__init__(min_fee)
        self._pre_volume_fee = pre_volume_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        return two_decimal_price(max(actual_volume * self.pre_volume_fee, self.min_fee))

    @property
    def pre_volume_fee(self):
        return self._pre_volume_fee


class ByTradeCommission(Commission):
    _pre_trade_fee = 0

    def __init__(self, pre_trade_fee: float = 0, min_fee: float = 0):
        super().__init__(min_fee)
        self._pre_trade_fee = pre_trade_fee

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        return two_decimal_price(max(self.pre_trade_fee, self.min_fee))

    @property
    def pre_trade_fee(self):
        return self._pre_trade_fee


class StampTaxCommission(FixedCommission):

    def __init__(self, tax_rate: float = 0, min_fee: float = 0):
        super().__init__(tax_rate, min_fee)

    def execute(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        if direction == OrderDirection.SELL:
            return two_decimal_price(max(actual_price * actual_volume * self.tax_rate, self.min_fee))
        else:
            return 0

    @property
    def tax_rate(self):
        return self._fee_rate
