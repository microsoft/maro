# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict, deque
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
        self.order_queue = deque()

        # Used to map from product id to slot index.
        self.product_index_mapping: Dict[int, int] = {}

        # What product we will carry.
        self.product_list = []

    def get_pending_order(self) -> Dict[int, int]:
        """Get orders that states is pending.

        Returns:
            dict: Dictionary of order that key is product id, value is quantity.
        """
        counter = defaultdict(int)

        for order in self.order_queue:
            counter[order.product_id] += order.quantity

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
                # product_index = self.product_index_mapping[order.product_id]
                # self.data_model.check_in_price[product_index] += order_total_price

                return order_total_price

        return 0

    def initialize(self):
        super(DistributionUnit, self).initialize()

        # Init product list in data model.
        index = 0
        for sku_id, sku in self.facility.skus.items():
            self.product_list.append(sku_id)

            self.product_index_mapping[sku_id] = index

            index += 1

        self._init_data_model()

        self.data_model.initialize(self.config.get("unit_price", 0))

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

        # NOTE: we moved delay_order_penalty from facility to sku, is this ok?
        # update order's delay penalty per tick.
        for order in self.order_queue:
            sku = self.facility.skus[order.product_id]
            product_index = self.product_index_mapping[order.product_id]

            #self.data_model.delay_order_penalty[product_index] += sku.delay_order_penalty
            self.data_model.delay_order_penalty[product_index] += self.facility.get_config("delay_order_penalty")

    def flush_states(self):
        for vehicle in self.vehicles:
            vehicle.flush_states()

    def reset(self):
        super(DistributionUnit, self).reset()

        self.order_queue.clear()
        self._init_data_model()

        # Reset vehicles.
        for vehicle in self.vehicles:
            vehicle.reset()

    def _init_data_model(self):
        for product_id in self.product_list:
            self.data_model.product_list.append(product_id)
            self.data_model.delay_order_penalty.append(0)
            self.data_model.check_in_price.append(0)
