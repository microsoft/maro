# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Counter, Deque, Dict, List, Optional, Union

from maro.simulator.scenarios.supply_chain.order import Order, OrderStatus
from maro.simulator.scenarios.supply_chain.units.storage import AddStrategy
from maro.simulator.scenarios.supply_chain.units.unitbase import BaseUnitInfo, UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


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

        self._init_statistics()

    def _init_statistics(self) -> None:
        # TODO：replace this with BE's event buffer
        self._order_on_the_way: Dict[int, List[Order]] = defaultdict(list)

        # A dict of pending order queue. Key: vehicle type; Value: pending order queue.
        self._order_queues: Dict[str, Deque[Order]] = defaultdict(deque)

        # Pending order related statistics.
        self._pending_order_number: int = 0
        self._total_pending_quantity: int = 0
        # TODO: could it be added to data model?
        # The key is product id, the value is accumulated requested product quantity.
        self.pending_product_quantity: Dict[int, int] = defaultdict(int)

        self._is_order_changed: bool = False

        # Only incremental action valid.
        self.total_order_num: int = 0
        self.finished_order_num: int = 0
        self.expired_order_num: int = 0
        self.actual_order_leading_time: Counter = Counter()
        self.order_schedule_delay_time: Counter = Counter()

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
            order.dest_facility.id in self.facility.downstream_vlt_infos[order.sku_id],
            order.vehicle_type in self.facility.downstream_vlt_infos[order.sku_id][order.dest_facility.id],
            order.required_quantity > 0
        ]):
            self._order_queues[order.vehicle_type].append(order)
            self._maintain_pending_order_info(order, is_increase=True)
            self.total_order_num += 1

            consumer = order.dest_facility.products[order.sku_id].consumer
            consumer.handle_order_successfully_placed(order)

            self.check_in_quantity_in_order[order.sku_id] += order.required_quantity
            sku = self.facility.skus[order.sku_id]
            order_total_price = sku.price * order.required_quantity  # TODO: add transportation cost or not?
            return order_total_price
        else:
            return 0

    def _try_load(self, sku_id: int, quantity: int) -> bool:
        """Try to load specified number of scheduled product.

        Args:
            quantity (int): Number to load.
        """
        return self.facility.storage.try_take_products({sku_id: quantity})

    def _try_unload(self, order: Order, tick: int) -> bool:
        """Try unload products into destination's storage."""
        unloaded = order.dest_facility.storage.try_add_products(
            {order.sku_id: order.pending_receive_quantity},
            add_strategy=AddStrategy.IgnoreUpperBoundAddInOrder,  # TODO: check which strategy to use.
        )

        # Update order if we unloaded any.
        if len(unloaded) > 0:
            assert len(unloaded) == 1
            unloaded_quantity = unloaded[order.sku_id]

            consumer = order.dest_facility.products[order.sku_id].consumer
            consumer.handle_order_received(order=order, received_quantity=unloaded_quantity, tick=tick)

        if order.order_status == OrderStatus.FINISHED:
            self.finished_order_num += 1
            self.actual_order_leading_time[tick - order.creation_tick] += 1

        return order.order_status == OrderStatus.FINISHED

    def _schedule_order(self, tick: int, order: Order) -> float:
        """Schedule order and return the daily transportation cost for this order.

        Args:
            tick (int): the system tick when calling this function.
            order (Order): the target order to schedule.

        Returns:
            float: the daily transportation cost for this order. Equals to unit_transportation_cost * payload.
        """
        vlt_info = self.facility.downstream_vlt_infos[order.sku_id][order.dest_facility.id][order.vehicle_type]

        arrival_tick = tick + vlt_info.vlt  # TODO: add random factor if needed

        # Update order states and add it into order_on_the_way.
        order.arrival_tick = arrival_tick
        order.add_payload(payload=order.required_quantity)
        order.unit_transportation_cost_per_day = vlt_info.unit_transportation_cost
        order.order_status = OrderStatus.ON_THE_WAY
        self._order_on_the_way[arrival_tick].append(order)
        self.order_schedule_delay_time[tick - order.creation_tick] += 1

        consumer = order.dest_facility.products[order.sku_id].consumer
        consumer.handle_order_scheduled(order, tick)

        return vlt_info.unit_transportation_cost * order.required_quantity

    def pre_step(self, tick: int) -> None:
        self.check_in_quantity_in_order.clear()
        self.transportation_cost.clear()
        self.delay_order_penalty.clear()

    def _maintain_pending_order_info(self, order: Order, is_increase: bool) -> None:
        indicator = 1 if is_increase else -1
        self._pending_order_number += 1 * indicator
        self._total_pending_quantity += order.required_quantity * indicator
        self.pending_product_quantity[order.sku_id] += order.required_quantity * indicator
        self._is_order_changed = True

    def try_schedule_orders(self, tick: int) -> None:
        for vehicle_type in self._vehicle_num.keys():
            order_load_failed: List[Order] = []

            for order in self._order_queues[vehicle_type]:
                # Check if the order still active.
                if order.expiration_buffer is not None and tick > order.creation_tick + order.expiration_buffer:
                    order.order_status = OrderStatus.EXPIRED
                    self._maintain_pending_order_info(order, is_increase=False)
                    self.expired_order_num += 1
                    # Update waiting order quantity info in Consumer.
                    consumer = order.dest_facility.products[order.sku_id].consumer
                    consumer.handle_order_expired(order)
                    continue
                # Try to schedule order and load products.
                if self._has_available_vehicle(vehicle_type) and self._try_load(order.sku_id, order.required_quantity):
                    transportation_cost_per_day = self._schedule_order(tick, order)
                    self.transportation_cost[order.sku_id] += transportation_cost_per_day
                    # The transportation cost of this newly scheduled order would be counted soon, do not count here.
                    self._busy_vehicle_num[vehicle_type] += 1
                    self._maintain_pending_order_info(order, is_increase=False)
                else:
                    # Count delay order penalty.
                    self.delay_order_penalty[order.sku_id] += self._unit_delay_order_penalty[order.sku_id]
                    order_load_failed.append(order)

            self._order_queues[vehicle_type] = order_load_failed

    def step(self, tick: int) -> None:
        # Update transportation cost for orders that are already on the way.
        # TODO: here do not distinguish on the way & pending unloading.
        for order_list in self._order_on_the_way.values():
            for order in order_list:
                self.transportation_cost[order.sku_id] += (
                    order.unit_transportation_cost_per_day * order.payload
                )

        # Schedule orders
        self.try_schedule_orders(tick)

    def handle_arrival_orders(self, tick: int) -> None:
        # Handle arrival payloads.
        for order in self._order_on_the_way[tick]:
            order.order_status = OrderStatus.PENDING_UNLOAD
            if self._try_unload(order, tick):
                self._busy_vehicle_num[order.vehicle_type] -= 1
            else:
                self._order_on_the_way[tick + 1].append(order)

        self._order_on_the_way.pop(tick)

    def flush_states(self) -> None:
        super(DistributionUnit, self).flush_states()

        if self._is_order_changed:
            self.data_model.pending_order_number = self._pending_order_number
            self.data_model.pending_product_quantity = self._total_pending_quantity
            self._is_order_changed = False

    def reset(self) -> None:
        super(DistributionUnit, self).reset()

        self.transportation_cost.clear()
        self.delay_order_penalty.clear()
        self.check_in_quantity_in_order.clear()

        for vehicle_type in self._vehicle_num.keys():
            self._busy_vehicle_num[vehicle_type] = 0

        self._reset_statistics()

    def _reset_statistics(self) -> None:
        self._order_on_the_way.clear()

        self._order_queues.clear()

        self._pending_order_number = 0
        self._total_pending_quantity = 0
        self.pending_product_quantity.clear()

        self._is_order_changed = False

        self.total_order_num = 0
        self.finished_order_num = 0
        self.expired_order_num = 0
        self.actual_order_leading_time = Counter()
        self.order_schedule_delay_time = Counter()

    def get_unit_info(self) -> DistributionUnitInfo:
        return DistributionUnitInfo(
            **super(DistributionUnit, self).get_unit_info().__dict__,
        )
