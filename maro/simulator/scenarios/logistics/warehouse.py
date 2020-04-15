from maro.simulator.frame import Frame, FrameNodeType

static_node = FrameNodeType.STATIC

class Warehouse:
    def __init__(self, index: int, stock:int, capacity:int, frame: Frame):
        self._index = index
        self._frame = frame
        self._id = id

        self._capacity = capacity # initial state
        self.capacity = capacity

        self._stock = stock # initial state

    @property
    def id(self):
        return self._id

    @property
    def index(self):
        return self._index

    @property
    def stock(self):
        return self._frame.get_attribute(static_node, self._index, "stock", 0)

    @stock.setter
    def stock(self, value: int):
        self._frame.set_attribute(static_node, self._index, "stock", 0, value)

    def fulfill_demand(self, value: int):
        self.stock = self.stock - value

    def supply_stock(self, value: int):
        self.stock = self.stock + value

    @property
    def demand(self):
        return self._frame.get_attribute(static_node, self._index, "demand", 0)

    @stock.setter
    def demand(self, value: int):
        self._frame.set_attribute(static_node, self._index, "demand", 0, value)

    def reset(self):
        self.stock = self._stock