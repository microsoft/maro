from maro.simulator.graph import Graph, GraphAttributeType

INT = GraphAttributeType.INT


def build(cell_num: int):
    graph = Graph(cell_num, 0)

    reg_attr = graph.register_attribute

    reg_attr("bikes", INT, 1)
    reg_attr("fulfillment", INT, 1)
    reg_attr("trip_requirement", INT, 1)
    reg_attr("shortage", INT, 1)
    reg_attr("capacity", INT, 1)

    # additional features
    # we split gendor into 3 fields
    reg_attr("unknown_gendors", INT, 1)
    reg_attr("males", INT, 1)
    reg_attr("females", INT, 1)
    reg_attr("weekday", INT, 1)

    # usertype
    reg_attr("subscriptor", INT, 1)
    reg_attr("customer", INT, 2)

    graph.setup()

    return graph