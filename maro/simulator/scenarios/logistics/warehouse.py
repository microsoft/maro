from maro.simulator.frame import Frame, FrameNodeType
from maro.simulator.scenarios.entity_base import EntityBase, IntAttribute, FloatAttribute, frame_node

static_node = FrameNodeType.STATIC

@frame_node(static_node)
class Warehouse(EntityBase):
    # this will register 2 int attributes in Frame, and same with previous property definition
    stock = IntAttribute()
    demand = IntAttribute()

    def __init__(self, index: int, stock:int, capacity:int, frame: Frame):
        super().__init__(frame, index)
        self._stock = stock # initial state
        self.capacity = capacity # this is not a frame attribute, so will not save into frame


    def fulfill_demand(self, value: int):
        self.stock -= value

    def supply_stock(self, value: int):
        self.stock += value

    def reset(self):
        self.stock = self._stock