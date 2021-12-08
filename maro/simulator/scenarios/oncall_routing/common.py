# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from .order import Order
from .route import Route


class Events(Enum):
    CARRIER_ARRIVAL = "carrier_arrival"
    ORDER_PROCESSING = "ready to process order"
    CARRIER_DEPARTURE = "carrier_departure"
    ONCALL_RECEIVE = "oncall_receive"


@dataclass
class Action:
    order_id: str
    route_name: str
    insert_index: int   # Insert before the i-th order of current remaining plan.
    in_segment_order: int = 0  # Relative order of multiple on-call orders with same insert_index.

    def __repr__(self) -> str:
        return (
            f"Action(order_id: {self.order_id}, route_name: {self.route_name}, "
            f"insert_index: {self.insert_index}, in_segment_order: {self.in_segment_order})"
        )


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


class OncallRoutingPayload(object):
    def __init__(
        self,
        get_oncall_orders_func: Callable[[], List[Order]],
        get_routes_info_func: Callable[[], List[Route]]
    ):
        self._get_oncall_orders_func: Callable[[], List[Order]] = get_oncall_orders_func
        self._get_routes_info_func: Callable[[], List[Route]] = get_routes_info_func

    @property
    def oncall_orders(self) -> List[Order]:
        return self._get_oncall_orders_func()

    @property
    def routes_info(self) -> List[Route]:
        # TODO: deep copy or?
        return self._get_routes_info_func()
