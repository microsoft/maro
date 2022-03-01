# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import Counter, defaultdict

from scipy.ndimage.interpolation import shift

from .. import ConsumerAction, ConsumerDataModel
from .extendunitbase import ExtendUnitBase
from .order import Order


class ConsumerUnit(ExtendUnitBase):
    """Consumer unit used to generate order to purchase from upstream by action."""

    def __init__(self):
        super(ConsumerUnit, self).__init__()

        self.open_orders = defaultdict(Counter)

        # States in python side.
        self.received = 0
        self.purchased = 0
        self.sources = []
        self.pending_order_daily = None
        self.order_product_cost = 0

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
        # New order for product.
        self.open_orders[source_id][product_id] += qty_delta

    def initialize(self):
        super(ConsumerUnit, self).initialize()

        self.pending_order_daily = [0] * self.world.configs.settings["pending_order_len"]

        sku = self.facility.skus[self.product_id]

        order_cost = self.facility.get_config("order_cost")
        assert isinstance(order_cost, int)

        assert isinstance(self.data_model, ConsumerDataModel)
        self.data_model.initialize(sku.price, order_cost)

        if self.facility.upstreams is not None:
            # Construct sources from facility's upstreams.
            sources = self.facility.upstreams.get(self.product_id, None)

            if sources is not None:
                # Is we are a supplier facility?
                is_supplier = getattr(self.parent, "manufacture", None) is not None

                # Current sku information.
                sku = self.world.get_sku_by_id(self.product_id)

                for source_facility in sources:
                    # We are a supplier unit, then the consumer is used to purchase source materials from upstreams.
                    # Try to find who will provide this kind of material.
                    if is_supplier:
                        if source_facility.products is not None:
                            for source_sku_id in sku.bom.keys():
                                if source_sku_id in source_facility.products:
                                    # This is a valid source facility.
                                    self.sources.append(source_facility.id)
                    else:
                        # If we are not a manufacturing, just check if upstream have this sku configuration.
                        if sku.id in source_facility.skus:
                            self.sources.append(source_facility.id)

    def step(self, tick: int):
        self._update_pending_order()

        assert self.action is None or isinstance(self.action, ConsumerAction)

        # NOTE: id == 0 means invalid,as our id is 1 based.
        if not self.action or self.action.quantity <= 0 or self.action.product_id <= 0 or self.action.source_id == 0:
            return

        # NOTE: we are using product unit as destination,
        # so we expect the action.source_id is and id of product unit
        self.update_open_orders(self.action.source_id, self.action.product_id, self.action.quantity)

        order = Order(self.facility, self.action.product_id, self.action.quantity, self.action.vlt)

        source_facility = self.world.get_facility_by_id(self.action.source_id)

        self.order_product_cost = source_facility.distribution.place_order(order)

        self.purchased = self.action.quantity

    def flush_states(self):
        assert self.action is None or isinstance(self.action, ConsumerAction)

        if self.received > 0:
            self.data_model.received = self.received
            self.data_model.total_received += self.received

        if self.purchased > 0:
            self.data_model.purchased = self.purchased
            self.data_model.total_purchased += self.purchased
            self.data_model.latest_consumptions = 1.0

        if self.order_product_cost > 0:
            self.data_model.order_product_cost = self.order_product_cost

        if self.action is not None and self.action.quantity > 0:
            self.data_model.order_quantity = self.action.quantity
            self.data_model.reward_discount = self.action.reward_discount

    def post_step(self, tick: int):
        assert self.action is None or isinstance(self.action, ConsumerAction)

        # Clear the action states per step.
        if self.action is not None:
            self.data_model.latest_consumptions = 0
            self.data_model.reward_discount = 0

            if self.action.quantity > 0:
                self.data_model.order_quantity = 0

        # This will set action to None.
        super(ConsumerUnit, self).post_step(tick)

        if self.received > 0:
            self.data_model.received = 0
            self.received = 0

        if self.purchased > 0:
            self.data_model.purchased = 0
            self.purchased = 0

        if self.order_product_cost > 0:
            self.data_model.order_product_cost = 0
            self.order_product_cost = 0

    def reset(self):
        super(ConsumerUnit, self).reset()

        self.pending_order_daily = [0] * self.world.configs.settings["pending_order_len"]

        self.open_orders.clear()

    def get_in_transit_quantity(self):
        quantity = 0

        for source_id, orders in self.open_orders.items():
            quantity += orders.get(self.product_id, 0)

        return quantity

    def _update_pending_order(self):
        self.pending_order_daily = shift(self.pending_order_daily, -1, cval=0)

    def get_unit_info(self):
        info = super().get_unit_info()

        info["sources"] = self.sources

        return info
