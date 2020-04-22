from maro.simulator.frame import FrameNodeType, Frame
from maro.simulator.scenarios.entity_base import frame_node, EntityBase, IntAttribute, FloatAttribute
from maro.simulator.scenarios.finance.reader import Stock as RawStock

STATIC_NODE = FrameNodeType.STATIC

@frame_node(STATIC_NODE)
class Stock(EntityBase):
    opening_price = FloatAttribute()
    closing_price = FloatAttribute()
    highest_price = FloatAttribute()
    lowest_price = FloatAttribute()
    trade_amount = FloatAttribute()
    daily_return = FloatAttribute(ndigits=4)

    trade_volume = IntAttribute()
    trade_num = IntAttribute()

    is_valid = IntAttribute()

    account_hold_num = IntAttribute()     # stock number that account hold

    def __init__(self, frame: Frame, index: int, code: str):
        super().__init__(frame, index)
        self._code = code

    @property
    def code(self):
        return self._code

    def reset(self):
        self.opening_price = 0
        self.closing_price = 0
        self.daily_return = 0
        self.highest_price = 0
        self.lowest_price = 0
        self.trade_amount = 0
        self.trade_num = 0
        self.trade_volume = 0
        self.is_valid = 0

    def fill(self, raw_stock: RawStock):
        self.opening_price = raw_stock.opening_price
        self.closing_price = raw_stock.closing_price
        self.daily_return = raw_stock.daily_return
        self.highest_price = raw_stock.daily_return
        self.lowest_price = raw_stock.lowest_price
        self.trade_amount = raw_stock.trade_amount
        self.trade_num = raw_stock.trade_num
        self.trade_volume = raw_stock.trade_volume
        self.is_valid = 1 if raw_stock.is_valid else 0