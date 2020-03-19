from maro.simulator.graph import Graph, ResourceNodeType

static_node = ResourceNodeType.STATIC

GENDOR_UNKNOWN = 0
GENDOR_MALE = 1
GENDOR_FEMALE = 2

USERTYPE_SUBSCRIPTOR = 0
USERTYPE_CUSTOMER = 1

class Station:
    def __init__(self, index: int, id: int, bikes: int, capacity:int, graph: Graph):
        self._index = index
        self._graph = graph
        self._id = id
        self._bikes = bikes
        self.capacity = capacity
        self.inventory = bikes

    @property
    def id(self):
        return self._id

    @property
    def index(self):
        return self._index

    @property
    def inventory(self):
        return self._graph.get_attribute(static_node, self._index, "inventory", 0)

    @inventory.setter
    def inventory(self, value: int):
        self._graph.set_attribute(static_node, self._index, "inventory", 0, value)

    @property
    def fullfillment(self):
        return self._graph.get_attribute(static_node, self._index, "fullfillment", 0)

    @property
    def orders(self):
        return self._graph.get_attribute(static_node, self._index, "orders", 0)

    @orders.setter
    def orders(self, value: int):
        self._graph.set_attribute(static_node, self._index, "orders", 0, value)

        self._update_fulfillment("fullfillment", value, self.shortage)

    @property
    def shortage(self):
        return self._graph.get_attribute(static_node, self._index, "shortage", 0)

    @shortage.setter
    def shortage(self, value: int):
        self._graph.set_attribute(static_node, self._index, "shortage", 0, value)

        self._update_fulfillment("fullfillment", self.orders, value)

    @property
    def acc_orders(self):
        return self._graph.get_attribute(static_node, self._index, "acc_orders", 0)

    @acc_orders.setter
    def acc_orders(self, value: int):
        self._graph.set_attribute(static_node, self._index, "acc_orders", 0, value)

        self._update_fulfillment("acc_fullfillment", value, self.orders)

    @property
    def acc_shortage(self):
        return self._graph.get_attribute(static_node, self._index, "acc_shortage", 0)

    @acc_shortage.setter
    def acc_shortage(self, value: int):
        self._graph.set_attribute(static_node, self._index, "acc_shortage", 0, value)

        self._update_fulfillment("acc_fullfillment", self.orders, value)

    @property
    def acc_fullfillment(self):
        return self._graph.get_attribute(static_node, self._index, "acc_fullfillment", 0)

    @property
    def capacity(self):
        return self._graph.get_attribute(static_node, self._index, "capacity", 0)

    @capacity.setter
    def capacity(self, value: int):
        self._graph.set_attribute(static_node, self._index, "capacity", 0, value)

    @property
    def unknow_gendors(self):
        return self._graph.get_attribute(static_node, self._index, "unknow_gendors", 0)

    @unknow_gendors.setter
    def unknow_gendors(self, value: int):
        self._graph.set_attribute(static_node, self._index, "unknow_gendors", 0, value)

    @property
    def males(self):
        return self._graph.get_attribute(static_node, self._index, "males", 0)

    @males.setter
    def males(self, value: int):
        self._graph.set_attribute(static_node, self._index, "males", 0, value)

    @property
    def females(self):
        return self._graph.get_attribute(static_node, self._index, "females", 0)

    @females.setter
    def females(self, value: int):
        self._graph.set_attribute(static_node, self._index, "females", 0, value)

    @property
    def weekday(self):
        return self._graph.get_attribute(static_node, self._index, "weekday", 0)

    @weekday.setter
    def weekday(self, value: int):
        self._graph.set_attribute(static_node, self._index, "weekday", 0, value)       

    @property
    def subscriptor(self):
        return self._graph.get_attribute(static_node, self._index, "subscriptor", 0)

    @subscriptor.setter
    def subscriptor(self, value: int):
        self._graph.set_attribute(static_node, self._index, "subscriptor", 0, value)       

    @property
    def customer(self):
        return self._graph.get_attribute(static_node, self._index, "customer", 0)

    @customer.setter
    def customer(self, value: int):
        self._graph.set_attribute(static_node, self._index, "customer", 0, value)     

    def update_gendor(self, gendor: int, num: int=1):
        if gendor == GENDOR_FEMALE:
            self.females += num
        elif gendor == GENDOR_MALE:
            self.males += num
        else:
            self.unknow_gendors += num

    def update_usertype(self, usertype: int, num: int = 1):
        if usertype == USERTYPE_SUBSCRIPTOR:
            self.subscriptor += num
        else:
            self.customer += num

    def reset(self):
        self.inventory = self._bikes
        self.shortage = 0
        self.orders = 0
        self.requirement = 0
        self.acc_orders = 0
        self.acc_shortage = 0
        self.unknow_gendors = 0
        self.males = 0
        self.females = 0
        self.weekday = 0
        self.subscriptor = 0 
        self.customer = 0

    def _update_fulfillment(self, field, orders, shortage):
        self._graph.set_attribute(static_node, self._index, field, 0, orders - shortage)