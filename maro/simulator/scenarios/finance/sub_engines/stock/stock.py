from maro.backends.frame import node, NodeBase, NodeAttribute


@node("stocks")
class Stock(NodeBase):
    opening_price = NodeAttribute("f")
    closing_price = NodeAttribute("f")
    highest_price = NodeAttribute("f")
    lowest_price = NodeAttribute("f")
    adj_closing_price = NodeAttribute("f")
    dividends = NodeAttribute("f")
    splits = NodeAttribute("f")

    trade_volume = NodeAttribute("i8")

    is_valid = NodeAttribute("i")

    account_hold_num = NodeAttribute("i")     # stock number that account hold
    average_cost = NodeAttribute("f")

    def __init__(self):
        pass

    def set_init_state(self, code: str):
        self._code = code

    @property
    def code(self):
        return self._code

    def reset(self):
        self.opening_price = 0
        self.closing_price = 0
        self.highest_price = 0
        self.lowest_price = 0
        self.adj_closing_price = 0
        self.dividends = 0
        self.splits = 0
        self.trade_volume = 0
        self.is_valid = 0

    def fill(self, raw_stock):
        self.opening_price = raw_stock.opening_price
        self.closing_price = raw_stock.closing_price
        self.highest_price = raw_stock.highest_price
        self.lowest_price = raw_stock.lowest_price
        self.adj_closing_price = raw_stock.adj_closing_price
        self.dividends = raw_stock.dividends
        self.splits = raw_stock.splits
        self.trade_volume = raw_stock.trade_volume
        self.is_valid = 1 #if raw_stock.is_valid else 0
