# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict, Counter

from .order import Order
from .skuunit import SkuUnit


class ConsumerUnit(SkuUnit):
    """Consumer unit used to generated order to purchase from upstream by action."""

    def __init__(self):
        super(ConsumerUnit, self).__init__()

        self.open_orders = defaultdict(Counter)

        # States in python side.
        self.received = 0
        self.purchased = 0
        self.order_cost = 0

    def on_order_reception(self, source_id: int, product_id: int, quantity: int, original_quantity: int):
        """Called after order product is received.

        Args:
            source_id (int): Where is the product from.
            product_id (int): What product we received.
            quantity (int): How many we received.
            original_quantity (int): How many we ordered.
        """
        self.received += quantity

        self.update_open_orders(source_id, product_id, -original_quantity)

    def update_open_orders(self, source_id: int, product_id: int, qty_delta: int):
        """Update the order states.

        Args:
            source_id (int): Where is the product from.
            product_id (int): What product in the order.
            qty_delta (int): Number of product to update (sum).
        """
        if qty_delta > 0:
            # New order for product.
            self.open_orders[source_id][product_id] += qty_delta
        else:
            # An order is completed, update the remaining number.
            self.open_orders[source_id][product_id] += qty_delta

    def initialize(self):
        super(ConsumerUnit, self).initialize()

    def step(self, tick: int):
        # NOTE: id == 0 means invalid,as our id is 1 based.
        if self.action is None or self.action.quantity <= 0 or self.action.consumer_product_id <= 0 or self.action.source_id == 0:
            return

        # NOTE: we are using product unit as destination,
        # so we expect the action.source_id is and id of product unit
        self.update_open_orders(self.action.source_id, self.action.consumer_product_id, self.action.quantity)

        order = Order(self.parent, self.action.consumer_product_id, self.action.quantity, self.action.vlt)

        source_facility = self.world.get_facility_by_id(self.action.source_id)

        self.order_cost = source_facility.distribution.place_order(order)

        self.purchased = self.action.quantity

    def flush_states(self):
        if self.received > 0:
            self.data_model.received = self.received
            self.data_model.total_received += self.received

        if self.purchased > 0:
            self.data_model.purchased = self.purchased
            self.data_model.total_purchased += self.purchased

        if self.order_cost > 0:
            self.data_model.order_product_cost = self.order_cost

    def post_step(self, tick: int):
        # Clear the action states per step.
        if self.action is not None:
            self.data_model.source_id = 0
            self.data_model.quantity = 0
            self.data_model.vlt = 0

        # This will set action to None.
        super(ConsumerUnit, self).post_step(tick)

        if self.received > 0:
            self.data_model.received = 0
            self.received = 0

        if self.purchased > 0:
            self.data_model.purchased = 0
            self.purchased = 0

        if self.order_cost > 0:
            self.data_model.order_product_cost = 0
            self.order_cost = 0

    def reset(self):
        super(ConsumerUnit, self).reset()

        self.open_orders.clear()

    def set_action(self, action: object):
        super(ConsumerUnit, self).set_action(action)

        # record the action
        self.data_model.source_id = action.source_id
        self.data_model.quantity = action.quantity
        self.data_model.vlt = action.vlt
