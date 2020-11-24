from maro.simulator.scenarios.finance.common import OrderDirection, two_decimal_price


class StockTradeCost():
    """Class for Calculating the commission and tax in finance scenario.

    Args:
        open_tax (float): Tax rate when buying a stock.
        close_tax (float): Tax rate when selling a stock.
        open_commission (float): Commission rate when buying a stock.
        close_commission (float): Commission rate when selling a stock.
        min_commission (float): minimum of the commission.
    """

    def __init__(
        self, open_tax: float, close_tax: float, open_commission: float, close_commission: float, min_commission: float
    ):
        self._open_tax = open_tax
        self._close_tax = close_tax
        self._open_commission = open_commission
        self._close_commission = close_commission
        self._min_commission = min_commission

    def calculate(self, direction: OrderDirection, actual_price: float, actual_volume: int) -> float:
        """Calculate the total cost of a trade.

        Args:
            direction (OrderDirection): The direction of the trade.
            actual_price (float): The executing price of the trade.
            actual_volume (int): The executing volume of the trade.
        """
        total_cost = 0
        if direction == OrderDirection.BUY:
            total_cost += max(actual_price * actual_volume * self._open_commission, self._min_commission)
            total_cost += actual_price * actual_volume * self._open_tax
        elif direction == OrderDirection.SELL:
            total_cost += max(actual_price * actual_volume * self._close_commission, self._min_commission)
            total_cost += actual_price * actual_volume * self._close_tax
        return two_decimal_price(total_cost)
