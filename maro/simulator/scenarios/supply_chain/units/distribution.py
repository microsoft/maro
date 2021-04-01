# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict, deque, Counter
from typing import Dict

from .order import Order
from .unitbase import UnitBase


class DistributionUnit(UnitBase):
    """Unit that used to receive and execute orders from downstream facilities.

    One distribution can accept all kind of sku order.
    """
    # Transport unit list of this distribution unit.
    vehicles = None

    def __init__(self):
        super().__init__()
        self.order_queue = deque()

        self.transportation_cost = Counter()
        self.delay_order_penalty = Counter()
        self.check_in_order = Counter()

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
                self.order_queue.append(order)

                order_total_price = sku.price * order.quantity

                # TODO: states related, enable it later if needed.
                self.check_in_order[order.product_id] += order.quantity

                return order_total_price

        return 0

    def initialize(self):
        super(DistributionUnit, self).initialize()

    def step(self, tick: int):
        for vehicle in self.vehicles:
            # If we have vehicle not on the way and there is any pending order
            if len(self.order_queue) > 0 and vehicle.quantity == 0:
                order = self.order_queue.popleft()

                # Schedule a job for available vehicle.
                # TODO: why vlt is determined by order?
                vehicle.schedule(
                    order.destination,
                    order.product_id,
                    order.quantity,
                    order.vlt
                )

            # Push vehicle.
            vehicle.step(tick)

            self.transportation_cost[vehicle.product_id] += abs(vehicle.step_reward)

            self.step_balance_sheet += vehicle.step_balance_sheet

        # update order's delay penalty per tick.
        for order in self.order_queue:
            self.delay_order_penalty[order.product_id] += self.facility.get_config("delay_order_penalty")

    def flush_states(self):
        for vehicle in self.vehicles:
            vehicle.flush_states()

        # TODO: optimize it later, only update if there is any changes
        self.data_model.remaining_order_quantity = sum(order.quantity for order in self.order_queue)
        self.data_model.remaining_order_number = len(self.order_queue)

    def reset(self):
        super(DistributionUnit, self).reset()

        self.order_queue.clear()

        # Reset vehicles.
        for vehicle in self.vehicles:
            vehicle.reset()

        self.transportation_cost.clear()
        self.check_in_order.clear()
        self.delay_order_penalty.clear()
