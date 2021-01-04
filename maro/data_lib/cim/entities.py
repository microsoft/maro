# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from enum import Enum

# item used to hold base value and related noise
NoisedItem = namedtuple("NoisedItem", ["index", "base", "noise"])

# data collection from data generator
CimDataCollection = namedtuple("CimDataCollection", [
    "total_containers",
    "past_stop_number",
    "future_stop_number",
    "cntr_volume",
    "order_mode",
    "ports_settings",
    "port_mapping",
    "vessels_settings",
    "vessel_mapping",
    "vessels_stops",
    "order_proportion",
    "routes",
    "route_mapping",
    "vessel_period_no_noise",
    "max_tick",
    "seed",
    "version"
])

# stop for vessel
Stop = namedtuple("Stop", [
    "index",
    "arrive_tick",
    "leave_tick",
    "port_idx",
    "vessel_idx"
])

# settings for port
PortSetting = namedtuple("PortSetting", [
    "index",
    "name",
    "capacity",
    "empty",
    "source_proportion",
    "target_proportions",
    "empty_return_buffer",
    "full_return_buffer"
])

# settings for vessel
VesselSetting = namedtuple("VesselSettings", [
    "index",
    "name",
    "capacity",
    "route_name",
    "start_port_name",
    "sailing_speed",
    "sailing_noise",
    "parking_duration",
    "parking_noise",
    "empty"
])

# a point in rote definition
RoutePoint = namedtuple("RoutePoint", [
    "index",
    "port_name",
    "distance"
])


class OrderGenerateMode(Enum):
    """Mode to generate orders from configuration.

    There are 2 modes for now:
    1. fixed: order is generated base on total containers, do not care about available empty container
    2. unfixed: order is generated with configured ratio, and considering available empty containers

    """
    FIXED = "fixed"
    UNFIXED = "unfixed"


# TODO: use object pooling to reduce memory cost
class Order:
    """
    Used to hold order information, this is for order generation.
    """
    summary_key = ["tick", "src_port_idx", "dest_port_idx", "quantity"]

    def __init__(self, tick: int, src_port_idx: int, dest_port_idx: int, quantity: int):
        """
        Create a new instants of order

        Args:
            tick (int): Generated tick of current order.
            src_port_idx (int): Source port of this order.
            dest_port_idx (int): Destination port id of this order.
            quantity (int): Container quantity of this order.
        """
        self.tick = tick
        self.src_port_idx = src_port_idx
        self.quantity = quantity
        self.dest_port_idx = dest_port_idx

    def __repr__(self):
        return (
            f"Order {{tick:{self.tick}, source port: {self.src_port_idx}, "
            f"dest port: {self.dest_port_idx} quantity: {self.quantity}}}"
        )
