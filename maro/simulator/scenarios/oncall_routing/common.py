from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, Union

from .order import Order

RouteNumber = Union[str, int]
OrderId = Union[str, int]


class Events(Enum):
    CARRIER_ARRIVAL = "carrier_arrival"
    ONCALL_RECEIVE = "oncall_receive"


class Coordinate(NamedTuple):
    lat: float
    lng: float


@dataclass
class Action:
    order_id: OrderId
    route_number: RouteNumber
    insert_index: int
    in_segment_order: int = 0


@dataclass
class OncallReceivePayload:
    orders: List[Order]


@dataclass
class CarrierArrivalPayload:
    route_number: RouteNumber


@dataclass
class PlanElement:
    order: Order
    est_arr_time: int  # Estimated arrival time
    act_arr_time: int  # Actual arrival time


@dataclass
class OncallRoutingPayload:
    oncall_orders: List[Order]
