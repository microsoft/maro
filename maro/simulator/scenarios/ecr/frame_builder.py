# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.frame import Frame, FrameNodeType

AT_STATIC = FrameNodeType.STATIC
AT_DYNAMIC = FrameNodeType.DYNAMIC
AT_GENERAL = FrameNodeType.GENERAL

int_attr = "i4"
float_attr = "f"

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
    register_attribute(AT_STATIC, "empty", int_attr)
    register_attribute(AT_STATIC, "full", int_attr)
    register_attribute(AT_STATIC, "early_discharge", int_attr)
    register_attribute(AT_STATIC, "capacity", float_attr)

    register_attribute(AT_DYNAMIC, "empty", int_attr)
    register_attribute(AT_DYNAMIC, "full", int_attr)
    register_attribute(AT_DYNAMIC, "early_discharge", int_attr)
    register_attribute(AT_DYNAMIC, "capacity", float_attr)
    register_attribute(AT_DYNAMIC, "remaining_space", int_attr)

    # attribute for port
    register_attribute(AT_STATIC,"on_shipper", int_attr)
    register_attribute(AT_STATIC,"on_consignee", int_attr)
    register_attribute(AT_STATIC,"booking", int_attr)
    register_attribute(AT_STATIC,"shortage", int_attr)
    register_attribute(AT_STATIC,"acc_booking", int_attr)
    register_attribute(AT_STATIC,"acc_shortage", int_attr)
    register_attribute(AT_STATIC,"fulfillment", int_attr)
    register_attribute(AT_STATIC,"acc_fulfillment", int_attr)

    # attribute for vessel

    # which route current vessel belongs to
    register_attribute(AT_DYNAMIC,"route_idx", int_attr, 1)

    # stop index in route, used to identify where is current vessel
    # last_loc_idx == next_loc_idx means vessel parking at a port
    register_attribute(AT_DYNAMIC,"last_loc_idx", int_attr, 1)
    register_attribute(AT_DYNAMIC,"next_loc_idx", int_attr, 1)

    # M previous stops (without current parking port)
    # this field should be update after arriving at a port
    register_attribute(AT_DYNAMIC,"past_stop_list", int_attr, stop_nums[0])

    # same with past_stop_list, but saves related ticks
    register_attribute(AT_DYNAMIC,"past_stop_tick_list", int_attr, stop_nums[0])

    register_attribute(AT_DYNAMIC,"future_stop_list", int_attr, stop_nums[1])
    register_attribute(AT_DYNAMIC,"future_stop_tick_list", int_attr, stop_nums[1])

    # distribution of full from port to port
    register_attribute(AT_GENERAL, "full_on_ports", int_attr, port_num * port_num)

    # distribution of full from vessel to port
    register_attribute(AT_GENERAL, "full_on_vessels", int_attr, vessel_num * port_num)

    # planed route info for vessels
    register_attribute(AT_GENERAL, "vessel_plans", int_attr, vessel_num * port_num)

    frame.setup()

    return frame