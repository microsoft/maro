from maro.backends.frame import NodeAttribute, NodeBase, node


@node("stocks")
class Stock(NodeBase):
    """Node to preserve stock quote information."""
    opening_price = NodeAttribute("f")
    closing_price = NodeAttribute("f")
    highest_price = NodeAttribute("f")
    lowest_price = NodeAttribute("f")
    adj_closing_price = NodeAttribute("f")
    dividends = NodeAttribute("f")
    splits = NodeAttribute("f")

    market_volume = NodeAttribute("i8")

    is_valid = NodeAttribute("i")

    account_hold_num = NodeAttribute("i")     # Stock volume that account holding.
    average_cost = NodeAttribute("f")

    def __init__(self):
        pass

    def set_init_state(self, code: str):
        """Set the initial information of the stock.

        Args:
            code (str): The code of the stock.
        """
        self._code = code

    @property
    def code(self):
        return self._code

    def reset(self):
        """Reset the is_valid property of the stock.
        So that the last available quote information still can be queried when the stock is not available to trade.
        """
        self.is_valid = 0

    def deep_reset(self):
        """Reset most of the properties of the stock."""
        self.reset()
        self.opening_price = 0
        self.closing_price = 0
        self.highest_price = 0
        self.lowest_price = 0
        self.adj_closing_price = 0
        self.dividends = 0
        self.splits = 0
        self.market_volume = 0

    def fill(self, raw_stock):
        """Fill the quote information from raw-stock to the stock instance."""
        self.opening_price = raw_stock.opening_price
        self.closing_price = raw_stock.closing_price
        self.highest_price = raw_stock.highest_price
        self.lowest_price = raw_stock.lowest_price
        self.adj_closing_price = raw_stock.adj_closing_price
        self.dividends = raw_stock.dividends
        self.splits = raw_stock.splits
        self.market_volume = raw_stock.market_volume
        self.is_valid = 1
