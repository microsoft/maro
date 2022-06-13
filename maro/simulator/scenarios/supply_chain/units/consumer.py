# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from maro.simulator.scenarios.supply_chain.actions import ConsumerAction
from maro.simulator.scenarios.supply_chain.datamodels import ConsumerDataModel
from maro.simulator.scenarios.supply_chain.order import Order, OrderStatus

from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class ConsumerUnitInfo(ExtendUnitInfo):
    source_facility_id_list: List[int]


class ConsumerUnit(ExtendUnitBase):
    """Consumer unit used to generate order to purchase from upstream by action."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(ConsumerUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )
        # States in python side.
        self._received: int = 0  # The quantity of product received in current step.

        self._purchased: int = 0  # The quantity of product that purchased from upstream.
        self._order_product_cost: float = 0  # order.quantity * upstream.price
        self._order_base_cost: float = 0  # order.quantity * unit_order_cost

        self.source_facility_id_list: List[int] = []

        self._unit_order_cost: float = 0

        self._init_statistics()

    def _init_statistics(self) -> None:
        self.order_quantity_on_the_way: Dict[int, int] = defaultdict(int)
        self._open_orders: Counter = Counter()

        # Dynamically changing statistics.
        self.pending_scheduled_order_quantity: int = 0
        self.in_transit_quantity: int = 0  # In transit = pending scheduled + on the way.

        # Only incremental action valid.
        self._total_order_num: int = 0
        self._finished_order_num: int = 0
        self._expired_order_num: int = 0
        self._actual_order_leading_time: Counter = Counter()
        self._order_schedule_delay_time: Counter = Counter()

    def get_order_statistics(self, tick: int) -> Dict[str, Union[int, Counter]]:
        return {
            # The accumulated order number that have been placed to distribution unit, = finished + expired + active.
            "total_order_num": self._total_order_num,
            # The accumulated finished order number (expired order number not included).
            "finished_order_num": self._finished_order_num,
            # The accumulated expired order number (expired due to waiting too long to load by upstream).
            "expired_order_num": self._expired_order_num,
            # The accumulated actual leading time distribution, leading time = finished time - order creation time.
            "actual_order_leading_time": self._actual_order_leading_time,
            # The accumulated scheduling delayed time distribution, delay time = departure time - order creation time.
            "order_schedule_delay_time": self._order_schedule_delay_time,
            # The current total pending scheduled (waiting to load by upstream) product quantity.
            "pending_scheduled_quantity": self.pending_scheduled_order_quantity,
            # The current active ordered product quantity, active = pending scheduled + on the way (+ pending unload).
            "active_ordered_quantity": self.in_transit_quantity,
            # The product quantity that will be received in a future time window.
            # The ones that not scheduled yet are not included here.
            "expected_future_received": self.get_pending_order_daily(tick),
        }

    def get_pending_order_daily(self, tick: int) -> List[int]:
        ret = [
            self.order_quantity_on_the_way[tick + i] for i in range(self.world.configs.settings["pending_order_len"])
        ]
        return ret

    def handle_order_successfully_placed(self, order: Order) -> None:
        self._open_orders[order.src_facility.id] += order.required_quantity

        self.pending_scheduled_order_quantity += order.required_quantity
        self.in_transit_quantity += order.required_quantity

        self._total_order_num += 1

    def handle_order_scheduled(self, order: Order, tick: int) -> None:
        # TODO: here the actual arrival tick is used.
        self.order_quantity_on_the_way[order.arrival_tick] += order.payload

        self.pending_scheduled_order_quantity -= order.required_quantity

        self._order_schedule_delay_time[tick - order.creation_tick] += 1

    def handle_order_expired(self, order: Order) -> None:
        self._open_orders[order.src_facility.id] -= order.required_quantity

        self.pending_scheduled_order_quantity -= order.required_quantity
        self.in_transit_quantity -= order.required_quantity

        self._expired_order_num += 1

    def handle_order_received(self, order: Order, received_quantity: int, tick: int) -> None:
        """Called after order product is received.

        Args:
            order(Order): The order the products received belongs to.
            received_quantity (int): How many we received.
            tick (int): The simulation tick.
        """
        assert order.sku_id == self.sku_id
        assert received_quantity > 0
        self._received += received_quantity

        order.receive(tick, received_quantity)

        # TODO: order quantity on the way
        self._open_orders[order.src_facility.id] -= received_quantity

        self.in_transit_quantity -= received_quantity

        if order.order_status == OrderStatus.FINISHED:
            self._finished_order_num += 1
            self._actual_order_leading_time[tick - order.creation_tick] += 1
        else:
            self.order_quantity_on_the_way[tick + 1] += order.pending_receive_quantity

    def initialize(self) -> None:
        super(ConsumerUnit, self).initialize()

        self._unit_order_cost = self.facility.skus[self.sku_id].unit_order_cost

        assert isinstance(self.data_model, ConsumerDataModel)

        self.data_model.initialize()

        self.source_facility_id_list = [
            source_facility.id for source_facility in self.facility.upstream_facility_list[self.sku_id]
        ]

    """
    ConsumerAction would be given after BE.step(), assume we take action at t0,
    the order would be placed to the DistributionUnit immediately (at t0)
    and would be scheduled if has available vehicle (at t0), assume in the case we have available vehicle,
    then the products would arrive at the destination at the end of (t0 + vlt) in the post_step(), which means
    at (t0 + vlt), these products can't be consumed to fulfill the demand from the downstreams/customer.
    """

    def process_actions(self, actions: List[ConsumerAction], tick: int) -> None:
        self._order_product_cost = self._order_base_cost = self._purchased = 0
        for action in actions:
            self._process_action(action, tick)

    def _process_action(self, action: ConsumerAction, tick: int) -> None:
        # NOTE: id == 0 means invalid, as our id is 1-based.
        if any([
            action.source_id not in self.source_facility_id_list,
            action.sku_id != self.sku_id,
            action.quantity <= 0,
        ]):
            return

        source_facility = self.world.get_facility_by_id(action.source_id)
        expected_vlt = source_facility.downstream_vlt_infos[self.sku_id][self.facility.id][action.vehicle_type].vlt

        order = Order(
            src_facility=source_facility,
            dest_facility=self.facility,
            sku_id=self.sku_id,
            quantity=action.quantity,
            vehicle_type=action.vehicle_type,
            creation_tick=tick,
            expected_finish_tick=tick + expected_vlt,
            expiration_buffer=action.expiration_buffer,
        )

        # Here the order cost is calculated by the upper distribution unit, with the sku price in that facility.
        self._order_product_cost += source_facility.distribution.place_order(order)
        # TODO: the order would be cancelled if there is no available vehicles,
        # TODO: but the cost is not decreased at that time.

        self._order_base_cost += order.required_quantity * self._unit_order_cost

        self._purchased += action.quantity

    def pre_step(self, tick: int) -> None:
        if self._received > 0:
            self._received = 0

            self.data_model.received = 0

        if self._purchased > 0:
            self._purchased = 0
            self._order_product_cost = 0
            self._order_base_cost = 0

            self.data_model.purchased = 0
            self.data_model.order_product_cost = 0
            self.data_model.order_base_cost = 0
            self.data_model.latest_consumptions = 0

    def step(self, tick: int) -> None:
        pass

    def post_step(self, tick: int) -> None:
        # TODO: cannot handle the case when unload failed.
        if tick in self.order_quantity_on_the_way:  # Remove data at tick to save storage
            self.order_quantity_on_the_way.pop(tick)

    def flush_states(self) -> None:
        if self._received > 0:
            self.data_model.received = self._received

        if self._purchased > 0:
            self.data_model.purchased = self._purchased
            self.data_model.order_product_cost = self._order_product_cost
            self.data_model.order_base_cost = self._order_base_cost
            self.data_model.latest_consumptions = 1.0

        if self._received > 0 or self._purchased > 0:
            self.data_model.in_transit_quantity = self.in_transit_quantity

    def reset(self) -> None:
        super(ConsumerUnit, self).reset()

        # Reset status in Python side.
        self._received = 0
        self._purchased = 0
        self._order_product_cost = 0
        self._order_base_cost = 0

        self._reset_statistics()

    def _reset_statistics(self) -> None:
        self.order_quantity_on_the_way.clear()
        self._open_orders.clear()

        self.pending_scheduled_order_quantity = 0
        self.in_transit_quantity = 0

        self._total_order_num = 0
        self._finished_order_num = 0
        self._expired_order_num = 0
        self._actual_order_leading_time.clear()
        self._order_schedule_delay_time.clear()

    def get_unit_info(self) -> ConsumerUnitInfo:
        return ConsumerUnitInfo(
            **super(ConsumerUnit, self).get_unit_info().__dict__,
            source_facility_id_list=self.source_facility_id_list,
        )
