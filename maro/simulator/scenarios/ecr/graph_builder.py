# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.graph import Graph, GraphAttributeType, GraphDataType

AT_STATIC = GraphAttributeType.STATIC_NODE
AT_DYNAMIC = GraphAttributeType.DYNAMIC_NODE
AT_GENERAL = GraphAttributeType.GENERAL

INT = GraphDataType.INT32
FLOAT = GraphDataType.FLOAT


def gen_ecr_graph(port_num: int, vessel_num: int, stop_nums: tuple):
    """
    Used to generated an ECR graph with predefined attributes

    Args:
        port_num (int): number of ports
        vessel_num (int): number of vessels
        stop_nums (tuple[int, int]): number of stops (past and future)

    Returns:
        Graph object that already setup
    """
    graph = Graph(port_num, vessel_num)

    register_attribute = graph.register_attribute

    # attributes for vessel and port
    register_attribute(AT_DYNAMIC, "empty", INT, 1)
    register_attribute(AT_STATIC, "empty", INT, 1)
    register_attribute(AT_DYNAMIC, "full", INT, 1)
    register_attribute(AT_STATIC, "full", INT, 1)
    register_attribute(AT_DYNAMIC, "capacity", FLOAT, 1)
    register_attribute(AT_STATIC, "capacity", FLOAT, 1)
    
    # attribute for port
    register_attribute(AT_STATIC, "on_shipper", INT, 1)
    register_attribute(AT_STATIC, "on_consignee", INT, 1)
    register_attribute(AT_STATIC, "booking", INT, 1)
    register_attribute(AT_STATIC, "shortage", INT, 1)
    register_attribute(AT_STATIC, "acc_booking", INT, 1)
    register_attribute(AT_STATIC, "acc_shortage", INT, 1)
    register_attribute(AT_STATIC, "fulfillment", INT, 1)
    register_attribute(AT_STATIC, "acc_fulfillment", INT, 1)

    # attribute for vessel
    register_attribute(AT_DYNAMIC, "early_discharge", INT, 1)
    register_attribute(AT_DYNAMIC, "remaining_space", INT, 1)

    # which route current vessel belongs to
    register_attribute(AT_DYNAMIC, "route_idx", INT, 1)

    # stop index in route, used to identify where is current vessel
    # last_loc_idx == next_loc_idx means vessel parking at a port
    register_attribute(AT_DYNAMIC, "last_loc_idx", INT, 1)
    register_attribute(AT_DYNAMIC, "next_loc_idx", INT, 1)

    # M previous stops (without current parking port)
    # this field should be update after arriving at a port
    register_attribute(AT_DYNAMIC, "past_stop_list", INT, stop_nums[0])

    # same with past_stop_list, but saves related ticks
    register_attribute(AT_DYNAMIC, "past_stop_tick_list", INT, stop_nums[0])

    register_attribute(AT_DYNAMIC, "future_stop_list", INT, stop_nums[1])
    register_attribute(AT_DYNAMIC, "future_stop_tick_list", INT, stop_nums[1])

    # distribution of full from port to port
    register_attribute(AT_GENERAL, "full_on_ports", INT, port_num * port_num)

    # distribution of full from vessel to port
    register_attribute(AT_GENERAL, "full_on_vessels", INT, vessel_num * port_num)

    # planed route info for vessels
    register_attribute(AT_GENERAL, "vessel_plans", INT, vessel_num * port_num)

    graph.setup()

    return graph
