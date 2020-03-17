from maro.simulator.graph import Graph, ResourceNodeType

static_node = ResourceNodeType.STATIC

class Station:
    def __init__(self, index: int, id: int, bikes: int, graph: Graph):
        self._index = index
        self._graph = graph
        self._id = id
        self._bikes = bikes

    @property
    def id(self):
        return self._id

    @property
    def index(self):
        return self._index

    @property
    def bikes(self):
        return self._graph.get_attribute(static_node, self._idx, "bike_num", 0)

    @bikes.setter
    def bikes(self, value: int):
        self._graph.set_attribute(static_node, self._idx, "bike_num", 0, value)

    @property
    def fullfillment(self):
        return self._graph.get_attribute(static_node, self._idx, "fullfillment", 0)

    @fullfillment.setter
    def fullfillment(self, value: int):
        self._graph.set_attribute(static_node, self._idx, "fullfillment", 0, value)

    @property
    def requirement(self):
        return self._graph.get_attribute(static_node, self._idx, "requirement", 0)

    @requirement.setter
    def requirement(self, value: int):
        self._graph.set_attribute(static_node, self._idx, "requirement", 0, value)

    @property
    def shortage(self):
        return self._graph.get_attribute(static_node, self._idx, "shortage", 0)

    @shortage.setter
    def shortage(self, value: int):
        self._graph.set_attribute(static_node, self._idx, "shortage", 0, value)

    def reset(self):
        self.bikes = self._bikes
        self.shortage = 0
        self.fullfillment = 0
        self.requirement = 0