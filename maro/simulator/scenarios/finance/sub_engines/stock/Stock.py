from maro.simulator.frame import FrameNodeType, Frame
from maro.simulator.scenarios.entity_base import build_frame, EntityBase, IntAttribute, FloatAttribute
from maro.simulator.scenarios.finance.reader import Stock as RawStock

STATIC_NODE = FrameNodeType.STATIC
DYNAMIC_NODE = FrameNodeType.DYNAMIC

class Stock(EntityBase):
    opening_price = FloatAttribute(STATIC_NODE)
    closing_price = FloatAttribute(STATIC_NODE)
    highest_price = FloatAttribute(STATIC_NODE)
    lowest_price = FloatAttribute(STATIC_NODE)
    trade_amount = FloatAttribute(STATIC_NODE)
    daily_return = FloatAttribute(STATIC_NODE, ndigits=4)

    trade_volume = IntAttribute(STATIC_NODE)
    trade_num = IntAttribute(STATIC_NODE)


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

    def fill(self, raw_stock: RawStock):
        self.opening_price = raw_stock.opening_price
        self.closing_price = raw_stock.closing_price
        self.daily_return = raw_stock.daily_return
        self.highest_price = raw_stock.daily_return
        self.lowest_price = raw_stock.lowest_price
        self.trade_amount = raw_stock.trade_amount
        self.trade_num = raw_stock.trade_num
        self.trade_volume = raw_stock.trade_volume