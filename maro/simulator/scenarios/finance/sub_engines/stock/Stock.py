from maro.simulator.frame import Frame, FrameNodeType
from functools import partial

STOCK_NODE = FrameNodeType.STATIC

class Stock:
    def __init__(self, frame: Frame, index: int, code: str):
        self._frame = frame
        self._index = index
        self._code = code

        self._getter_partial = partial(self._frame.get_attribute, node_type=STOCK_NODE, node_id=self._index, attribute_index=0)
        self._setter_partial = partial(self._frame.set_attribute, node_type=STOCK_NODE, node_id=self._index, attribute_index=0)
    @property
    def index(self):
        return self._index

    @property
    def opening_price(self):
        return self._getter_partial(attribute_name="openning_price")

    @opening_price.setter
    def opening_price(self, value):
        self._setter_partial(attribute_name="opening_price")