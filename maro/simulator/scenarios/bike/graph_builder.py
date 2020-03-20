from maro.simulator.graph import Graph, GraphAttributeType

INT32 = GraphAttributeType.INT


def build(station_num: int):
    graph = Graph(station_num, 0)

    reg_attr = graph.register_attribute

    reg_attr("bikes", INT32, 1)
    reg_attr("fullfillment", INT32, 1)
    reg_attr("orders", INT32, 1)
    reg_attr("shortage", INT32, 1)
    reg_attr("capacity", INT32, 1)
    reg_attr("acc_orders", INT32, 1)
    reg_attr("acc_shortage", INT32, 1)
    reg_attr("acc_fullfillment", INT32, 1)
    
    # additional features
    # we split gendor into 3 fields
    reg_attr("unknow_gendors", INT32, 1)
    reg_attr("males", INT32, 1)
    reg_attr("females", INT32, 1)
    reg_attr("weekday", INT32, 1)

    # usertype
    reg_attr("subscriptor", INT32, 1)
    reg_attr("customer", INT32, 2)

    graph.setup()

    return graph