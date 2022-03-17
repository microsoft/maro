# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import Counter, defaultdict, deque
from typing import Dict, List

from .order import Order
from .unitbase import UnitBase
from .vehicle import VehicleUnit


class DistributionUnit(UnitBase):
    """Unit that used to receive and execute orders from downstream facilities.

    One distribution can accept all kind of sku order.
    """
    # Vehicle unit list of this distribution unit.
    vehicles: List[VehicleUnit] = None

    def __init__(self):
        super().__init__()
        self.order_queue = deque()

        self.transportation_cost = Counter()
        self.delay_order_penalty = Counter()
        self.check_in_order = Counter()

        self.base_delay_order_penalty = 0

        self._is_order_changed = False

    def get_pending_order(self) -> Dict[int, int]:
        """Get orders that states is pending.

        Returns:
            dict: Dictionary of order that key is product id, value is quantity.
        """
        counter = defaultdict(int)

        for order in self.order_queue:
            counter[order.product_id] += order.quantity

        for vehicle in self.vehicles:
            if vehicle.is_enroute():
                counter[vehicle.product_id] += (vehicle.requested_quantity - vehicle.payload)

        return counter

    def place_order(self, order: Order) -> int:
        """Place an order in the pending queue.

        Args:
            order (Order): Order to insert.

        Returns:
            int: Total price of this order.
        """
        if order.quantity > 0:
            sku = self.facility.skus[order.product_id]

            if sku is not None:
                self._is_order_changed = True

                self.order_queue.append(order)

                order_total_price = sku.price * order.quantity

                self.check_in_order[order.product_id] += order.quantity

                return order_total_price

        return 0

    def initialize(self):
        super(DistributionUnit, self).initialize()

        self.base_delay_order_penalty = self.facility.get_config("delay_order_penalty", 0)

    def _step_impl(self, tick: int):
        for vehicle in self.vehicles:
            # If we have vehicle not on the way and there is any pending order.
            if len(self.order_queue) > 0 and vehicle.requested_quantity == 0:
                order = self.order_queue.popleft()

                # Schedule a job for available vehicle.
                # TODO: why vlt is determined by order?
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
        for order in self.order_queue:
            self.delay_order_penalty[order.product_id] += self.base_delay_order_penalty

    def flush_states(self):
        super(DistributionUnit, self).flush_states()

        for vehicle in self.vehicles:
            vehicle.flush_states()

        if self._is_order_changed:
            self._is_order_changed = False

            self.data_model.remaining_order_quantity = sum(order.quantity for order in self.order_queue)
            self.data_model.remaining_order_number = len(self.order_queue)

    def reset(self):
        super(DistributionUnit, self).reset()

        # Reset status in Python side.
        self.order_queue.clear()

        self.transportation_cost.clear()
        self.delay_order_penalty.clear()
        self.check_in_order.clear()

        self._is_order_changed = False

        # Reset vehicles.
        for vehicle in self.vehicles:
            vehicle.reset()
