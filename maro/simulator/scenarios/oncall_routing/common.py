# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .order import Order


class Events(Enum):
    CARRIER_ARRIVAL = "carrier_arrival"
    ORDER_PROCESSING = "ready to process order"
    CARRIER_DEPARTURE = "carrier_departure"
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


@dataclass
class CarrierDeparturePayload:
    carrier_idx: int


@dataclass
class OrderProcessingPayload:
    carrier_idx: int


@dataclass
class PlanElement:
    order: Order
    estimated_duration_from_last: Optional[int] = None  # Estimated duration from last stop
    actual_duration_from_last: Optional[int] = None  # Actual duration from last stop


@dataclass
class OncallRoutingPayload:
    oncall_orders: List[Order]
