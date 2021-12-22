# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from .duration_time_predictor import EstimatedDurationPredictor
from .order import Order


class Events(Enum):
    CARRIER_ARRIVAL = "carrier_arrival"
    ORDER_PROCESSING = "ready to process order"
    CARRIER_DEPARTURE = "carrier_departure"
    ONCALL_RECEIVE = "oncall_receive"


class Action(object):
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id

    def __repr__(self) -> str:
        return "%s {order_id: %s}" % (self.__class__.__name__, self.order_id)


class PostponeAction(Action):
    def __init__(self, order_id: str) -> None:
        super().__init__(order_id)


class AllocateAction(Action):
    def __init__(self, order_id: str, route_name: str, insert_index: int, in_segment_order: int = 0) -> None:
        super().__init__(order_id)
        self.route_name = route_name
        self.insert_index = insert_index  # Insert before the i-th order of current remaining plan.
        self.in_segment_order = in_segment_order  # Relative order of multiple on-call orders with same insert_index.

    def __repr__(self) -> str:
        return "%s {order_id: %s, route_name: %s, insert_index: %d, in_segment_order: %d}" % (
            self.__class__.__name__, self.order_id, self.route_name, self.insert_index, self.in_segment_order
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
        get_route_plan_dict_func: Callable[[], Dict[str, List[Order]]],
        get_estimated_duration_predictor_func: Callable[[], EstimatedDurationPredictor],
        get_route_meta_info_dict: Callable[[], Dict[str, dict]]
    ):
        self._get_oncall_orders_func: Callable[[], List[Order]] = get_oncall_orders_func
        self._get_route_plan_dict_func: Callable[[], Dict[str, List[Order]]] = get_route_plan_dict_func
        self._get_estimated_duration_predictor_func = get_estimated_duration_predictor_func
        self._get_route_meta_info_dict = get_route_meta_info_dict

        self._oncall_orders_cache: Optional[List[Order]] = None
        self._route_plan_dict_cache: Optional[Dict[str, List[Order]]] = None
        self._estimated_duration_predictor_cache: Optional[EstimatedDurationPredictor] = None
        self._route_meta_info_dict_cache: Optional[Dict[str, dict]] = None

    @property
    def oncall_orders(self) -> List[Order]:
        if self._oncall_orders_cache is None:
            self._oncall_orders_cache = self._get_oncall_orders_func()
        return self._oncall_orders_cache

    @property
    def route_plan_dict(self) -> Dict[str, List[Order]]:
        if self._route_plan_dict_cache is None:
            self._route_plan_dict_cache = self._get_route_plan_dict_func()
        return self._route_plan_dict_cache

    @property
    def estimated_duration_predictor(self) -> EstimatedDurationPredictor:
        if self._estimated_duration_predictor_cache is None:
            self._estimated_duration_predictor_cache = self._get_estimated_duration_predictor_func()
        return self._estimated_duration_predictor_cache

    @property
    def route_meta_info_dict(self) -> Dict[str, dict]:
        if self._route_meta_info_dict_cache is None:
            self._route_meta_info_dict_cache = self._get_route_meta_info_dict()
        return self._route_meta_info_dict_cache
