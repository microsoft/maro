# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import deque, defaultdict
from typing import Dict

from .order import Order
from .unitbase import UnitBase


class DistributionUnit(UnitBase):
    """Unit that used to receive and execute orders from downstream facilities.

    One distribution can accept all kind of sku order.
    """
    # Transport unit list of this distribution unit.
    transports = None

    def __init__(self):
        self.order_queue = deque()

        # Used to map from product id to slot index.
        self.product_index_mapping: Dict[int, int] = {}

        self.product_list = []

    def get_pending_order(self):
        """Get orders that states is pending.

        Returns:
            dict: Dictionary of order that key is product id, value is quantity.
        """
        counter = defaultdict(int)

        for order in self.order_queue:
            counter[order.product_id] += order.quantity

        return counter

    def place_order(self, order: Order):
        """Place an order in the pending queue.

        Args:
            order (Order): Order to insert.
        """
        if order.quantity > 0:
            sku = self.facility.skus[order.product_id]

            if sku is not None:
                self.order_queue.append(order)

                order_total_price = sku.price * order.quantity

                # TODO: states related, enable it later if needed.
                # product_index = self.product_index_mapping[order.product_id]
                # self.data.check_in_price[product_index] += order_total_price

                return order_total_price

        return 0

    def initialize(self):
        index = 0

        # Init product list in data model.
        for sku_id, sku in self.facility.skus.items():
            self.product_list.append(sku_id)

            self.data_model.product_list.append(sku_id)
            self.data_model.delay_order_penalty.append(0)
            self.data_model.check_in_price.append(0)

            self.product_index_mapping[sku_id] = index

            index += 1

    def step(self, tick: int):
        for vehicle in self.transports:
            # If we have vehicle not on the way and there is any pending order
            if len(self.order_queue) > 0 and vehicle.location == 0:
                order = self.order_queue.popleft()

                # schedule a job for vehicle
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

            self.data_model.delay_order_penalty[product_index] += sku.delay_order_penalty

    def reset(self):
        super(DistributionUnit, self).reset()

        self.order_queue.clear()

        for product_id in self.product_list:
            self.data_model.product_list.append(product_id)
            self.data_model.delay_order_penalty.append(0)
            self.data_model.check_in_price.append(0)
