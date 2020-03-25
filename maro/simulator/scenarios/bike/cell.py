from maro.simulator.frame import Frame, FrameNodeType

static_node = FrameNodeType.STATIC

GENDOR_UNKNOWN = 0
GENDOR_MALE = 1
GENDOR_FEMALE = 2

USERTYPE_SUBSCRIPTOR = 0
USERTYPE_CUSTOMER = 1

HOLIDAY = 0
WORKDAY = 1

CELL_BEIGHBOR_NUM = 6

class Cell:
    def __init__(self, index: int, capacity:int, bikes: int, frame: Frame):
        self._index = index
        self._frame = frame
        self._id = id
        self._bikes = bikes
        self._capacity = capacity
        self.capacity = capacity
        self.bikes = bikes

        self._neighbors_cache = None # since the neighbors will not change, so we can keep it here

    @property
    def id(self):
        return self._id

    @property
    def index(self):
        return self._index

    @property
    def bikes(self):
        return self._frame.get_attribute(static_node, self._index, "bikes", 0)

    @bikes.setter
    def bikes(self, value: int):
        self._frame.set_attribute(static_node, self._index, "bikes", 0, value)

    @property
    def fulfillment(self):
        return self._frame.get_attribute(static_node, self._index, "fulfillment", 0)

    @property
    def trip_requirement(self):
        return self._frame.get_attribute(static_node, self._index, "trip_requirement", 0)

    @trip_requirement.setter
    def trip_requirement(self, value: int):
        self._frame.set_attribute(static_node, self._index, "trip_requirement", 0, value)

        self._update_fulfillment("fulfillment", value, self.shortage)

    @property
    def shortage(self):
        return self._frame.get_attribute(static_node, self._index, "shortage", 0)

    @shortage.setter
    def shortage(self, value: int):
        self._frame.set_attribute(static_node, self._index, "shortage", 0, value)

        self._update_fulfillment("fulfillment", self.trip_requirement, value)

    @property
    def capacity(self):
        return self._frame.get_attribute(static_node, self._index, "capacity", 0)

    @capacity.setter
    def capacity(self, value: int):
        self._frame.set_attribute(static_node, self._index, "capacity", 0, value)

    @property
    def unknown_gendors(self):
        return self._frame.get_attribute(static_node, self._index, "unknown_gendors", 0)

    @unknown_gendors.setter
    def unknown_gendors(self, value: int):
        self._frame.set_attribute(static_node, self._index, "unknown_gendors", 0, value)

    @property
    def males(self):
        return self._frame.get_attribute(static_node, self._index, "males", 0)

    @males.setter
    def males(self, value: int):
        self._frame.set_attribute(static_node, self._index, "males", 0, value)

    @property
    def females(self):
        return self._frame.get_attribute(static_node, self._index, "females", 0)

    @females.setter
    def females(self, value: int):
        self._frame.set_attribute(static_node, self._index, "females", 0, value)

    @property
    def weekday(self):
        return self._frame.get_attribute(static_node, self._index, "weekday", 0)

    @weekday.setter
    def weekday(self, value: int):
        self._frame.set_attribute(static_node, self._index, "weekday", 0, value)       

    @property
    def subscriptor(self):
        return self._frame.get_attribute(static_node, self._index, "subscriptor", 0)

    @subscriptor.setter
    def subscriptor(self, value: int):
        self._frame.set_attribute(static_node, self._index, "subscriptor", 0, value)       

    @property
    def customer(self):
        return self._frame.get_attribute(static_node, self._index, "customer", 0)

    @customer.setter
    def customer(self, value: int):
        self._frame.set_attribute(static_node, self._index, "customer", 0, value)   

    @property
    def temperature(self):
        return self._frame.get_attribute(static_node, self._index, "temperature", 0)

    @temperature.setter
    def temperature(self, value: float):
        self._frame.set_attribute(static_node, self._index, "temperature", 0, value)

    @property
    def weather(self):
        return self._frame.get_attribute(static_node, self._index, "weather", 0)

    @weather.setter
    def weather(self, value: int):
        self._frame.set_attribute(static_node, self._index, "weather" ,0, value)

    @property
    def holiday(self):
        val = self._frame.get_attribute(static_node, self._index, "holiday", 0)

        return val == HOLIDAY

    @holiday.setter
    def holiday(self, value: bool):
        self._frame.set_attribute(static_node, self._index, "holiday", 0, HOLIDAY if value else WORKDAY)  

    @property
    def extra_cost(self):
        return self._frame.get_attribute(static_node, self._index, "extra_cost", 0)

    @extra_cost.setter
    def extra_cost(self, value):
        self._frame.set_attribute(static_node, self._index, "extra_cost", 0, value)

    @property
    def neighbors(self):
        # here we use cached neighbors to speedup
        return self._neighbors_cache

    def set_neighbors(self, neighbors: list):
        self._neighbors_cache = []

        for i, cell_idx in enumerate(neighbors):
      
            self._neighbors_cache.append(cell_idx)
            self._frame.set_attribute(static_node, self._index, "neighbors", i, cell_idx)

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
        self.shortage = 0
        self.trip_requirement = 0
        self.unknown_gendors = 0
        self.males = 0
        self.females = 0
        self.weekday = 0
        self.subscriptor = 0 
        self.customer = 0
        self.holiday = WORKDAY

        self.set_neighbors(self._neighbors_cache)

    def _update_fulfillment(self, field, trip_requirement, shortage):
        self._frame.set_attribute(static_node, self._index, field, 0, trip_requirement - shortage)