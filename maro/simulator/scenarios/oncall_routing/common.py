# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

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
    carrier_idx: int
    # TODO: carrier name, route name, detailed stop info?


@dataclass
class PlanElement:
    order: Order
    estimated_duration_from_last: Optional[int] = None  # Estimated duration from last stop
    actual_duration_from_last: Optional[int] = None  # Actual duration from last stop


@dataclass
class OncallRoutingPayload:
    oncall_orders: List[Order]
