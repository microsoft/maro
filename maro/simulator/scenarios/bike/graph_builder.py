from maro.simulator.graph import Graph, GraphAttributeType

int_attr = GraphAttributeType.INT


def build(station_num: int):
    graph = Graph(station_num, 0)

    reg_attr = graph.register_attribute

    reg_attr("bike_num", int_attr, 1)
    reg_attr("fullfillment", int_attr, 1)
    reg_attr("requirement", int_attr, 1)
    reg_attr("shortage", int_attr, 1)

    graph.setup()

    return graph