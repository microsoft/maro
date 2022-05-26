# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class request_column(Enum):
    vessel_header = "tick, index, capacity, empty, full, remaining_space, route_idx"
    decision_header = "tick, vessel_index, port_index, quantity"
    order_header = "tick, from_port_index, dest_port_index, quantity"
    port_header = "tick, index, capacity, empty, full, shortage, booking, fulfillment"


class request_settings(Enum):
    request_url = "http://127.0.0.1:9000/exec"
    request_header = {
        "content-type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "PUT,GET,POST,DELETE,OPTIONS",
        "Cache-Control": "no-cache, no-transform",
    }
