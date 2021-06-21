# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple

# settings for port
RealPortSetting = namedtuple("RealPortSetting", [
    "index",
    "name",
    "capacity",
    "empty",
    "empty_return_buffer",
    "full_return_buffer"
])

OrderTuple = namedtuple("Order", [
    "tick",
    "src_port_idx",
    "dest_port_idx",
    "quantity"
])

# data collection from data loader
CimRealDataCollection = namedtuple("CimRealDataCollection", [
    "past_stop_number",
    "future_stop_number",
    "container_volume",
    "ports_settings",
    "port_mapping",
    "vessels_settings",
    "vessel_mapping",
    "routes",
    "route_mapping",
    "vessel_period_without_noise",
    "vessels_stops",
    "orders",
    "max_tick",
    "seed"
])
