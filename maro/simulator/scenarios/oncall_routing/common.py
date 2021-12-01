# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import List

from .order import Order


class Events(Enum):
    CARRIER_ARRIVAL = "carrier_arrival"
    ONCALL_RECEIVE = "oncall_receive"


@dataclass
class Action:
    order_id: str
    route_name: str
    insert_index: int
    in_segment_order: int = 0


@dataclass
class OncallReceivePayload:
    orders: List[Order]


@dataclass
class CarrierArrivalPayload:
    route_name: str


@dataclass
class PlanElement:
    order: Order
    est_arr_time: int  # Estimated arrival time
    act_arr_time: int  # Actual arrival time


@dataclass
class OncallRoutingPayload:
    oncall_orders: List[Order]
