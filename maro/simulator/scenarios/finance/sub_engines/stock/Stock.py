from maro.simulator.frame import FrameNodeType, Frame
from maro.simulator.scenarios.modelbase import build_frame, ModelBase, IntAttribute, FloatAttribute

STATIC_NODE = FrameNodeType.STATIC
DYNAMIC_NODE = FrameNodeType.DYNAMIC

class Stock(ModelBase):
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