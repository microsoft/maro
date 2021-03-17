# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import warnings
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
        self.sources = []

    def on_order_reception(self, source_id: int, product_id: int, quantity: int, original_quantity: int):
        """Called after order product is received.

        Args:
            source_id (int): Where is the product from (facility id).
            product_id (int): What product we received.
            quantity (int): How many we received.
            original_quantity (int): How many we ordered.
        """
        self.received += quantity

        self.update_open_orders(source_id, product_id, -original_quantity)

    def update_open_orders(self, source_id: int, product_id: int, qty_delta: int):
        """Update the order states.

        Args:
            source_id (int): Where is the product from (facility id).
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

        if self.facility.upstreams is not None:
            # Construct sources from facility's upstreams.
            sources = self.facility.upstreams.get(self.product_id, None)

            if sources is not None:
                # Is we are a supplier facility?
                is_supplier = self.parent.manufacture is not None

                # Current sku information.
                sku = self.world.get_sku_by_id(self.product_id)

                for source_facility in sources:
                    # We are a supplier unit, then the consumer is used to purchase source materials from upstreams.
                    # Try to find who will provide this kind of material.
                    if is_supplier:
                        if source_facility.products is not None:
                            warnings.warn(f"Invalid upstream configuration for sku: {sku.id}.")

                            for source_sku_id in sku.bom.keys():
                                if source_sku_id in source_facility.products:
                                    # This is a valid source facility.
                                    self.sources.append(source_facility.id)
                    else:
                        # If we are not a manufacturing, just check if upstream have this sku configuration.
                        if sku.id in source_facility.skus:
                            self.sources.append(source_facility.id)

            self._init_data_model()

    def step(self, tick: int):
        # We must have a source to purchase.
        if len(self.sources) == 0:
            warnings.warn(f"No sources for consumer: {self.id}, sku: {self.product_id} in facility: {self.facility.id}")
            return

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

        self._init_data_model()

    def set_action(self, action: object):
        super(ConsumerUnit, self).set_action(action)

        # record the action
        self.data_model.source_id = action.source_id
        self.data_model.quantity = action.quantity
        self.data_model.vlt = action.vlt

    def _init_data_model(self):
        for source in self.sources:
            self.data_model.sources.append(source)
