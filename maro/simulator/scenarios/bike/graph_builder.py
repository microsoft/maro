from maro.simulator.graph import Graph, GraphAttributeType

int_attr = GraphAttributeType.INT


def build(station_num: int):
    graph = Graph(station_num, 0)

    reg_attr = graph.register_attribute

    reg_attr("inventory", int_attr, 1)
    reg_attr("fullfillment", int_attr, 1)
    reg_attr("orders", int_attr, 1)
    reg_attr("shortage", int_attr, 1)
    reg_attr("gendor", int_attr, 3)
    reg_attr("weekday", int_attr, 1)
    reg_attr("usertype", int_attr, 3)
    reg_attr("capacity", int_attr, 1)

    graph.setup()

    return graph