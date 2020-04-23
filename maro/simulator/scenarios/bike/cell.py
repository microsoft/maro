from maro.simulator.frame import Frame, FrameNodeType
from maro.simulator.scenarios.entity_base import EntityBase, IntAttribute, FloatAttribute, frame_node

STATIC_NODE = FrameNodeType.STATIC

GENDOR_UNKNOWN = 0
GENDOR_MALE = 1
GENDOR_FEMALE = 2

USERTYPE_SUBSCRIPTOR = 0
USERTYPE_CUSTOMER = 1

HOLIDAY = 0
WORKDAY = 1

CELL_BEIGHBOR_NUM = 6

@frame_node(STATIC_NODE)
class Cell(EntityBase):
    bikes = IntAttribute()
    capacity = IntAttribute()
    fulfillment = IntAttribute()
    trip_requirement = IntAttribute()
    shortage = IntAttribute()
    unknown_gendors = IntAttribute()
    males = IntAttribute()
    females = IntAttribute()
    weekday = IntAttribute()
    subscriptor = IntAttribute()
    customer = IntAttribute()
    temperature = IntAttribute()
    weather = IntAttribute()
    holiday = IntAttribute()
    extra_cost = IntAttribute()
    neighbors = IntAttribute(slot_num = CELL_BEIGHBOR_NUM)

    def __init__(self, index: int, capacity: int, bikes: int, frame: Frame):
        super().__init__(frame, index)

        self._bikes = bikes
        self._capacity = capacity

        self._neighbors_cache = None # since the neighbors will not change, so we can keep it here

    @property
    def neighbor_list(self):
        # here we use cached neighbors to speedup
        return self._neighbors_cache

    def set_neighbors(self, neighbors: list):
        self._neighbors_cache = []

        for i, cell_idx in enumerate(neighbors):
      
            self._neighbors_cache.append(cell_idx)
            self.neighbors[i] = cell_idx

    def update_gendor(self, gendor: int, num: int=1):
        if gendor == GENDOR_FEMALE:
            self.females += num
        elif gendor == GENDOR_MALE:
            self.males += num
        else:
            self.unknown_gendors += num

    def update_usertype(self, usertype: int, num: int = 1):
        if usertype == USERTYPE_SUBSCRIPTOR:
            self.subscriptor += num
        else:
            self.customer += num

    def reset(self):
        self.capacity = self._capacity
        self.bikes = self._bikes

        self.set_neighbors(self._neighbors_cache)

    # auto bind as callback when related attribute value changed
    def _on_trip_requirement_changed(self, slot_index: int, new_value):
        self._update_fulfillment(new_value, self.shortage)

    def _on_shortage_changed(self, slot_index: int, new_value):
        self._update_fulfillment(self.trip_requirement, new_value)

    def _update_fulfillment(self, trip_requirement: int, shortage: int):
        self.fulfillment = trip_requirement - shortage