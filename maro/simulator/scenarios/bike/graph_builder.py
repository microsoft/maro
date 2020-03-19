from maro.simulator.graph import Graph, GraphAttributeType

int_attr = GraphAttributeType.INT


def build(station_num: int):
    graph = Graph(station_num, 0)

    reg_attr = graph.register_attribute

    reg_attr("inventory", int_attr, 1)
    reg_attr("fullfillment", int_attr, 1)
    reg_attr("orders", int_attr, 1)
    reg_attr("shortage", int_attr, 1)
    reg_attr("capacity", int_attr, 1)
    reg_attr("acc_orders", int_attr, 1)
    reg_attr("acc_shortage", int_attr, 1)
    reg_attr("acc_fullfillment", int_attr, 1)
    
    # we split gendor into 3 fields
    reg_attr("unknow_gendors", int_attr, 1)
    reg_attr("males", int_attr, 1)
    reg_attr("females", int_attr, 1)
    reg_attr("weekday", int_attr, 1)

    # usertype
    reg_attr("subscriptor", int_attr, 1)
    reg_attr("customer", int_attr, 2)

    graph.setup()

    return graph