# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from maro.simulator.scenarios.supply_chain.order import Order
from .unitbase import UnitBase, BaseUnitInfo
from .vehicle import VehicleStatus, VehicleUnit

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class DistributionUnitInfo(BaseUnitInfo):
    vehicle_node_index_list: List[int]


class DistributionUnit(UnitBase):
    """Unit that used to receive and execute orders from downstream facilities.

    One distribution can accept all kind of sku order.
    """
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(DistributionUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

        # Vehicle unit dict of this distribution unit. Key: vehicle type; Value: a list of vehicle instances.
        self.vehicles: Dict[str, List[VehicleUnit]] = defaultdict(list)

        # A dict of pending order queue. Key: vehicle type; Value: pending order queue.
        self._order_queues: Dict[str, deque] = defaultdict(deque)

        # The transportation cost of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination here.
        self.transportation_cost = Counter()
        # The delay penalty of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.delay_order_penalty = Counter()
        # The check-in product quantity in orders of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.check_in_quantity_in_order = Counter()

        self._unit_delay_order_penalty: Dict[int, float] = {}

        self._is_order_changed: bool = False

    def get_pending_product_quantities(self) -> Dict[int, int]:
        """Count the requested product quantity in pending orders. Only used by BE metrics.

        Returns:
            Dict[int, int]: The key is product id, the value is accumulated requested product quantity.
        """
        counter = defaultdict(int)

        for order_queue in self._order_queues.values():
            for order in order_queue:
                counter[order.product_id] += order.quantity

        for vehicle_list in self.vehicles.values():  # TODO: check whether count these quantity in or not
            for vehicle in vehicle_list:
                if vehicle.status != VehicleStatus.Free:
                    counter[vehicle.product_id] += (vehicle.requested_quantity - vehicle.payload)

        return counter

    def place_order(self, order: Order) -> int:
        """Place an order in the pending order queue, and calculate the corresponding order fee.

        Args:
            order (Order): Order to insert.

        Returns:
            int: The corresponding total order fee, will paid by the consumer.
        """
        if order.quantity > 0:
            sku = self.facility.skus[order.product_id]

            if sku is not None:
                self._is_order_changed = True

                self._order_queues[order.vehicle_type].append(order)

                order_total_price = sku.price * order.quantity

                self.check_in_quantity_in_order[order.product_id] += order.quantity

                return order_total_price

        return 0

    def initialize(self):
        super(DistributionUnit, self).initialize()

        for product_id in self.facility.products.keys():
            self._unit_delay_order_penalty[product_id] = self.facility.skus[product_id].unit_delay_order_penalty

    def _step_impl(self, tick: int):
        # TODO: update vehicle types and distribution step logic
        for vehicle_type, vehicle_list in self.vehicles.items():
            for vehicle in vehicle_list:
                # If we have vehicle not on the way and there is any pending order.
                if len(self._order_queues[vehicle_type]) > 0 and vehicle.requested_quantity == 0:
                    order: Order = self._order_queues[vehicle_type].popleft()

                    # Schedule a job for available vehicle.
                    vehicle.schedule(
                        order.destination,
                        order.product_id,
                        order.quantity,
                        order.vlt,
                    )

                    self._is_order_changed = True

                # Push vehicle.
                vehicle.step(tick)

                self.transportation_cost[vehicle.product_id] += abs(vehicle.cost)

        # Update order's delay penalty per tick.
        for order_queue in self._order_queues.values():
            for order in order_queue:
                self.delay_order_penalty[order.product_id] += self._unit_delay_order_penalty[order.product_id]

    def flush_states(self):
        super(DistributionUnit, self).flush_states()

        for vehicle_list in self.vehicles.values():
            for vehicle in vehicle_list:
                vehicle.flush_states()

        if self._is_order_changed:
            self._is_order_changed = False

            self.data_model.pending_product_quantity = sum(
                order.quantity
                for order_queue in self._order_queues.values()
                for order in order_queue
            )
            self.data_model.pending_order_number = sum(
                len(order_queue)
                for order_queue in self._order_queues.values()
            )

    def reset(self):
        super(DistributionUnit, self).reset()

        # Reset status in Python side.
        self._order_queues.clear()

        self.transportation_cost.clear()
        self.delay_order_penalty.clear()
        self.check_in_quantity_in_order.clear()

        self._is_order_changed = False

        # Reset vehicles.
        for vehicle_list in self.vehicles.values():
            for vehicle in vehicle_list:
                vehicle.reset()

    def get_unit_info(self) -> DistributionUnitInfo:
        return DistributionUnitInfo(
            **super(DistributionUnit, self).get_unit_info().__dict__,
            vehicle_node_index_list=[vehicle.data_model_index for vehicle in self.children],
        )
