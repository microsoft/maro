# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import collections
import typing
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from maro.simulator.scenarios.supply_chain.order import Order

from maro.simulator.scenarios.supply_chain.units.storage import AddStrategy
from maro.simulator.scenarios.supply_chain.units.unitbase import BaseUnitInfo, UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class DistributionPayload:
    arrival_tick: int
    order: Order
    unit_transportation_cost_per_day: float
    payload: int


@dataclass
class DistributionUnitInfo(BaseUnitInfo):
    pass


class DistributionUnit(UnitBase):
    """Unit that used to receive and execute orders from downstream facilities.

    One distribution can accept all kind of sku order.
    """

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(DistributionUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

        self._vehicle_num: Dict[str, Optional[int]] = {}
        self._busy_vehicle_num: Dict[str, int] = {}

        # TODO：replace this with BE's event buffer
        self._payload_on_the_way: Dict[int, List[DistributionPayload]] = defaultdict(list)
        self._order_quantity_on_the_way: Dict[int, List[Payload]] = defaultdict(list)

        # A dict of pending order queue. Key: vehicle type; Value: pending order queue.
        self._order_queues: Dict[str, deque] = defaultdict(deque)
        self._pending_order_number: int = 0
        self._total_pending_quantity: int = 0
        self._pending_product_quantity: Dict[int, int] = defaultdict(int)
        self._is_order_changed: bool = False
        # self.pending_order_daily: Optional[List[int]] = None

        # Below 3 attributes are used to track states for ProductUnit's data model
        # The transportation cost of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination here.
        self.transportation_cost: Dict[int, float] = defaultdict(float)
        # The delay penalty of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.delay_order_penalty: Dict[int, float] = defaultdict(float)
        # The check-in product quantity in orders of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.check_in_quantity_in_order: Dict[int, float] = defaultdict(float)

        self._unit_delay_order_penalty: Dict[int, float] = {}

    def initialize(self) -> None:
        super(DistributionUnit, self).initialize()

        for vehicle_type, vehicle_config in self.config.items():
            vehicle_num = None
            if vehicle_config["number"] is not None:
                vehicle_num = int(vehicle_config["number"])
            self._vehicle_num[vehicle_type] = vehicle_num
            self._busy_vehicle_num[vehicle_type] = 0

            # TODO: add vehicle patient setting if needed

        for sku_id in self.facility.products.keys():
            self._unit_delay_order_penalty[sku_id] = self.facility.skus[sku_id].unit_delay_order_penalty

    def _has_available_vehicle(self, vehicle_type: str) -> bool:
        return (
            self._vehicle_num[vehicle_type] is None
            or self._busy_vehicle_num[vehicle_type] < self._vehicle_num[vehicle_type]
        )

    def place_order(self, order: Order) -> float:
        """Place an order in the pending order queue, and calculate the corresponding order fee.

        Args:
            order (Order): Order to be inserted.

        Returns:
            float: The corresponding total order fee, will paid by the consumer.
        """
        # TODO: to indicate whether it is a valid order or not in Return value?
        if all([
            order.sku_id in self.facility.downstream_vlt_infos,
            order.destination.id in self.facility.downstream_vlt_infos[order.sku_id],
            order.vehicle_type in self.facility.downstream_vlt_infos[order.sku_id][order.destination.id],
            order.quantity > 0
        ]):
            self._order_queues[order.vehicle_type].append(order)
            self._maintain_pending_order_info(order, departure=False)

            self.check_in_quantity_in_order[order.sku_id] += order.quantity
            sku = self.facility.skus[order.sku_id]
            order_total_price = sku.price * order.quantity  # TODO: add transportation cost or not?
            return order_total_price
        else:
            return 0

    @property
    def pending_product_quantity(self) -> Dict[int, int]:  # TODO: add it into data model.
        """Count the requested product quantity in pending orders. Only used by BE metrics.

        Returns:
            Dict[int, int]: The key is product id, the value is accumulated requested product quantity.
        """
        return self._pending_product_quantity

    def _try_load(self, sku_id: int, quantity: int) -> bool:
        """Try to load specified number of scheduled product.

        Args:
            quantity (int): Number to load.
        """
        return self.facility.storage.try_take_products({sku_id: quantity})

    def _try_unload(self, payload: DistributionPayload ) -> bool:
        """Try unload products into destination's storage."""
        unloaded = payload.order.destination.storage.try_add_products(
            {payload.order.sku_id: payload.payload},
            add_strategy=AddStrategy.IgnoreUpperBoundAddInOrder,  # TODO: check which strategy to use.
        )

        # Update order if we unloaded any.
        if len(unloaded) > 0:
            assert len(unloaded) == 1
            unloaded_quantity = unloaded[payload.order.sku_id]

            payload.order.destination.products[payload.order.sku_id].consumer.on_order_reception(
                source_id=self.facility.id,
                sku_id=payload.order.sku_id,
                received_quantity=unloaded_quantity,
                required_quantity=payload.order.quantity,
            )

            payload.payload -= unloaded_quantity

        return payload.payload == 0

    def _schedule_order(self, tick: int, order: Order) -> float:
        """Schedule order and return the daily transportation cost for this order.

        Args:
            tick (int): the system tick when calling this function.
            order (Order): the target order to schedule.

        Returns:
            float: the daily transportation cost for this order. Equals to unit_transportation_cost * payload.
        """
        vlt_info = self.facility.downstream_vlt_infos[order.sku_id][order.destination.id][order.vehicle_type]

        arrival_tick = tick + vlt_info.vlt  # TODO: add random factor if needed

        self._payload_on_the_way[arrival_tick].append(DistributionPayload(
            arrival_tick=arrival_tick,
            order=order,
            unit_transportation_cost_per_day=vlt_info.unit_transportation_cost,
            payload=order.quantity,
        ))

        dest_consumer = order.destination.products[order.sku_id].consumer
        if vlt_info.vlt < len(dest_consumer.pending_order_daily):  # Check use (arrival tick - tick) or vlt
            dest_consumer.pending_order_daily[vlt_info.vlt] += order.quantity

        else:
            self._order_quantity_on_the_way[arrival_tick] += order.quantity
        return vlt_info.unit_transportation_cost * order.quantity

    def pre_step(self, tick: int) -> None:
        self.check_in_quantity_in_order.clear()
        self.transportation_cost.clear()
        self.delay_order_penalty.clear()

    def _maintain_pending_order_info(self, order: Order, departure: bool) -> None:
        indicator = 1 if departure else -1
        self._pending_order_number -= 1 * indicator
        self._total_pending_quantity -= order.quantity * indicator
        self._pending_product_quantity[order.sku_id] -= order.quantity * indicator
        self._is_order_changed = True

    def try_schedule_orders(self, tick: int) -> None:
        for vehicle_type in self._vehicle_num.keys():
            # Schedule if there are available vehicles
            order_load_failed: List[Order] = []
            while all([
                len(self._order_queues[vehicle_type]) > 0,
                self._has_available_vehicle(vehicle_type),
            ]):
                order: Order = self._order_queues[vehicle_type].popleft()
                if self._try_load(order.sku_id, order.quantity):
                    transportation_cost_per_day = self._schedule_order(tick, order)
                    self.transportation_cost[order.sku_id] += transportation_cost_per_day

                    # The transportation cost of this newly scheduled order would be counted soon, do not count here.
                    self._busy_vehicle_num[vehicle_type] += 1
                    self._maintain_pending_order_info(order, departure=True)
                else:
                    order_load_failed.append(order)
            self._order_queues[vehicle_type].extend(order_load_failed)

            # Else count delay order penalty
            for order in self._order_queues[vehicle_type]:
                self.delay_order_penalty[order.sku_id] += self._unit_delay_order_penalty[order.sku_id]

    def step(self, tick: int) -> None:
        # Update transportation cost for orders that are already on the way.

        for payload_list in self._payload_on_the_way.values():
            for payload in payload_list:
                self.transportation_cost[payload.order.sku_id] += (
                    payload.unit_transportation_cost_per_day * payload.payload
                )

        # Schedule orders
        self.try_schedule_orders(tick)

    def handle_arrival_payloads(self, tick: int) -> None:
        # Handle arrival payloads.
        for payload in self._payload_on_the_way[tick]:
            if self._try_unload(payload):
                self._busy_vehicle_num[payload.order.vehicle_type] -= 1
            else:
                self._payload_on_the_way[tick + 1].append(payload)

        self._payload_on_the_way.pop(tick)

    def flush_states(self) -> None:
        super(DistributionUnit, self).flush_states()

        if self._is_order_changed:
            self.data_model.pending_order_number = self._pending_order_number
            self.data_model.pending_product_quantity = self._total_pending_quantity
            self._is_order_changed = False

    def pending_order_daily(self, tick) -> list:

        pending_order_daily : Dict[int, List[int]]=defaultdict(list)


        for payload_list in self._payload_on_the_way.values():
            for payload in payload_list:
                    arrival_tick = payload.arrival_tick
                    order = payload.order
                    vlt_info = self.facility.downstream_vlt_infos[order.sku_id][order.destination.id][order.vehicle_type]
                    sku_id = order.sku_id
                    lens = self.world.configs.settings["pending_order_len"]
                    if len(pending_order_daily) == 0:
                        pending_order_daily[sku_id] = [0] * self.world.configs.settings["pending_order_len"]

                    if vlt_info.vlt < self.world.configs.settings['pending_order_len']:  # Check use (arrival tick - tick) or vlt
                        pending_order_daily[order.sku_id][(arrival_tick - tick) % vlt_info.vlt] = order.quantity
                    else:
                        if (tick + lens) >= payload.arrival_tick:
                            pending_order_daily[order.sku_id][lens - (tick % lens) -1] += self._order_quantity_on_the_way[tick + self.lens]
                            # del self._order_quantity_on_the_way[tick + self.world.configs.settings['pending_order_len'] - 1]
        return pending_order_daily

    def reset(self) -> None:
        super(DistributionUnit, self).reset()

        # Reset status in Python side.
        self._order_queues.clear()
        self._pending_order_number = 0
        self._total_pending_quantity = 0
        self._pending_product_quantity.clear()
        self._is_order_changed = False

        self.transportation_cost.clear()
        self.delay_order_penalty.clear()
        self.check_in_quantity_in_order.clear()

        for vehicle_type in self._vehicle_num.keys():
            self._busy_vehicle_num[vehicle_type] = 0
        self._payload_on_the_way.clear()
        self._order_quantity_on_the_way.clear()

    def get_unit_info(self) -> DistributionUnitInfo:
        return DistributionUnitInfo(
            **super(DistributionUnit, self).get_unit_info().__dict__,
        )
