from maro.simulator.frame import Frame, FrameNodeType
from maro.simulator.scenarios.entity_base import EntityBase, IntAttribute, FloatAttribute, frame_node


static_node = FrameNodeType.STATIC


@frame_node(static_node)
class Warehouse(EntityBase):
    # this will register int attributes in Frame, and same with previous property definition
    weekday = IntAttribute()
    stock = IntAttribute()
    demand = IntAttribute()
    fulfilled = IntAttribute()
    unfulfilled = IntAttribute()

    def __init__(self, index: int, initial_stock: int, max_capacity: int, frame: Frame):
        super().__init__(frame, index)
        self.index = index
        self.name = "warehouse_{}".format(index)
        self._initial_stock = initial_stock # initial state
        self.max_capacity = max_capacity # this is not a frame attribute, so will not save into frame

    def fulfill_demand(self, tick: int, value: int):
        self.weekday = tick % 7
        self.demand = value
        self.fulfilled = min(self.stock, value)
        self.unfulfilled = max(0, value - self.stock)
        self.stock -= self.fulfilled

    def supply_stock(self, value: int):
        self.stock += value

    def reset(self):
        self.stock = self._initial_stock
        self.fulfilled = 0
        self.unfulfilled = 0
