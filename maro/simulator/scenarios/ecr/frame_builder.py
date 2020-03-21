# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.frame import Frame, FrameAttributeType

int_attr = FrameAttributeType.INT
float_attr = FrameAttributeType.FLOAT
mat_attr = FrameAttributeType.INT_MAT


def gen_ecr_frame(port_num: int, vessel_num: int, stop_nums: tuple):
    """
    Used to generated an ECR frame with predefined attributes

    Args:
        port_num (int): number of ports
        vessel_num (int): number of vessels
        stop_nums (tuple[int, int]): number of stops (past and future)

    Returns:
        Frame object that already setup
    """
    frame = Frame(port_num, vessel_num)

    register_attribute = frame.register_attribute

    # attributes for vessel and port
    register_attribute("empty", int_attr, 1)
    register_attribute("full", int_attr, 1)
    register_attribute("early_discharge", int_attr, 1)
    register_attribute("capacity", float_attr, 1)
    register_attribute("remaining_space", float_attr, 1)

    # attribute for port
    register_attribute("on_shipper", int_attr, 1)
    register_attribute("on_consignee", int_attr, 1)
    register_attribute("booking", int_attr, 1)
    register_attribute("shortage", int_attr, 1)
    register_attribute("acc_booking", int_attr, 1)
    register_attribute("acc_shortage", int_attr, 1)
    register_attribute("fulfillment", int_attr, 1)
    register_attribute("acc_fulfillment", int_attr, 1)

    # attribute for vessel

    # which route current vessel belongs to
    register_attribute("route_idx", int_attr, 1)

    # stop index in route, used to identify where is current vessel
    # last_loc_idx == next_loc_idx means vessel parking at a port
    register_attribute("last_loc_idx", int_attr, 1)
    register_attribute("next_loc_idx", int_attr, 1)

    # M previous stops (without current parking port)
    # this field should be update after arriving at a port
    register_attribute("past_stop_list", int_attr, stop_nums[0])

    # same with past_stop_list, but saves related ticks
    register_attribute("past_stop_tick_list", int_attr, stop_nums[0])

    register_attribute("future_stop_list", int_attr, stop_nums[1])
    register_attribute("future_stop_tick_list", int_attr, stop_nums[1])

    # distribution of full from port to port
    register_attribute("full_on_ports", mat_attr, port_num * port_num, port_num, port_num)

    # distribution of full from vessel to port
    register_attribute("full_on_vessels", mat_attr, vessel_num * port_num, vessel_num, port_num)

    # planed route info for vessels
    register_attribute("vessel_plans", mat_attr, vessel_num * port_num, vessel_num, port_num)

    frame.setup()

    return frame
