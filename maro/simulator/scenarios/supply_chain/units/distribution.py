# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .order import Order
from .unitbase import UnitBase, BaseUnitInfo
from .vehicle import VehicleUnit

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

        # Vehicle unit list of this distribution unit.
        self.vehicles: List[VehicleUnit] = None

        # The pending order queue.
        self._order_queue = deque()

        # The transportation cost of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination here.
        self.transportation_cost = Counter()
        # The delay penalty of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.delay_order_penalty = Counter()
        # The check-in product quantity in orders of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.check_in_quantity_in_order = Counter()

        self._base_delay_order_penalty: float = 0

        self._is_order_changed: bool = False

    def get_pending_product_quantities(self) -> Dict[int, int]:
        """Count the requested product quantity in pending orders. Only used by BE metrics.

        Returns:
            Dict[int, int]: The key is product id, the value is accumulated requested product quantity.
        """
        counter = defaultdict(int)

        for order in self._order_queue:
            counter[order.product_id] += order.quantity

        for vehicle in self.vehicles:
            if vehicle.is_enroute():
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

                self._order_queue.append(order)

                order_total_price = sku.price * order.quantity  # TODO: checking the meaning of the sku.price here.

                self.check_in_quantity_in_order[order.product_id] += order.quantity

                return order_total_price

        return 0

    def initialize(self):
        super(DistributionUnit, self).initialize()

        self._base_delay_order_penalty = self.facility.get_config("delay_order_penalty", 0)

    def _step_impl(self, tick: int):
        for vehicle in self.vehicles:
            # If we have vehicle not on the way and there is any pending order.
            if len(self._order_queue) > 0 and vehicle.requested_quantity == 0:
                order: Order = self._order_queue.popleft()

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
        for order in self._order_queue:
            self.delay_order_penalty[order.product_id] += self._base_delay_order_penalty  # TODO: here consider the product id only, not consider the destination.

    def flush_states(self):
        super(DistributionUnit, self).flush_states()

        for vehicle in self.vehicles:
            vehicle.flush_states()

        if self._is_order_changed:
            self._is_order_changed = False

            self.data_model.pending_product_quantity = sum(order.quantity for order in self._order_queue)
            self.data_model.pending_order_number = len(self._order_queue)

    def reset(self):
        super(DistributionUnit, self).reset()

        # Reset status in Python side.
        self._order_queue.clear()

        self.transportation_cost.clear()
        self.delay_order_penalty.clear()
        self.check_in_quantity_in_order.clear()

        self._is_order_changed = False

        # Reset vehicles.
        for vehicle in self.vehicles:
            vehicle.reset()

    def get_unit_info(self) -> DistributionUnitInfo:
        return DistributionUnitInfo(
            **super(DistributionUnit, self).get_unit_info().__dict__,
            vehicle_node_index_list=[vehicle.data_model_index for vehicle in self.children],
        )
