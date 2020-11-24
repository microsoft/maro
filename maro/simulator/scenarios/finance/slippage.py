from maro.simulator.scenarios.finance.common import OrderDirection, two_decimal_price


class Slippage():
    """Base class for calculating the slippage of an order."""

    def __init__(self):
        pass

    def calculate(self, order_direction: OrderDirection, order_volume: int, order_price: float, market_volume: int):
        pass


class FixedSlippage(Slippage):
    """Fixed slippage has a fixed slippage from the expected price.

    Args:
        slippage_rate (float): Slippage rate of the slippage.
    """

    def __init__(self, slippage_rate: float = 0):
        super().__init__()
        self._slippage_rate = slippage_rate

    def calculate(
        self, order_direction: OrderDirection, order_volume: int, order_price: float, market_volume: int
    ) -> float:
        """Calculate the slippage price of the order.

        Args:
            order_direction (OrderDirection): The direction of the order.
            order_volume (int): The expected trade volume of the order.
            order_price (float): The expected trade price of the order.
            market_volume (int): The trade volume of the stock in the market during the tick.

        Returns:
            float: The actual price with the slippage.
        """
        delta_price = order_price * self._slippage_rate / 2
        if order_direction == OrderDirection.BUY:
            actual_price = order_price + delta_price
        else:
            actual_price = order_price - delta_price
        return two_decimal_price(actual_price)
