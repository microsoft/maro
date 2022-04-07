# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

from scipy.ndimage.interpolation import shift

from maro.simulator.scenarios.supply_chain.actions import ConsumerAction
from maro.simulator.scenarios.supply_chain.datamodels import ConsumerDataModel

from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .order import Order


@dataclass
class ConsumerUnitInfo(ExtendUnitInfo):
    source_facility_id_list: List[int]


class ConsumerUnit(ExtendUnitBase):
    """Consumer unit used to generate order to purchase from upstream by action."""

    def __init__(self):
        super(ConsumerUnit, self).__init__()

        self._open_orders = defaultdict(Counter)

        # States in python side.
        self._received: int = 0  # The quantity of product received in current step.
        self._purchased: int = 0  # The quantity of product that purchased from upperstream.
        self._order_product_cost: float = 0
        self.source_facility_id_list: List[int] = []
        self.pending_order_daily: Optional[List[int]] = None

    def on_order_reception(self, source_id: int, product_id: int, received_quantity: int, required_quantity: int):
        """Called after order product is received.

        Args:
            source_id (int): Where is the product from (facility id).
            product_id (int): What product we received.
            quantity (int): How many we received.
            original_quantity (int): How many we ordered.
        """
        self._received += received_quantity

        self.update_open_orders(source_id, product_id, -required_quantity)

    def update_open_orders(self, source_id: int, product_id: int, additional_quantity: int):
        """Update the order states.

        Args:
            source_id (int): Where is the product from (facility id).
            product_id (int): What product in the order.
            qty_delta (int): Number of product to update (sum).
        """
        # New order for product.
        self._open_orders[source_id][product_id] += additional_quantity

    def initialize(self):
        super(ConsumerUnit, self).initialize()

        self.pending_order_daily = [0] * self.world.configs.settings["pending_order_len"]

        sku = self.facility.skus[self.product_id]

        # TODO: check is the order cost the basis cost for each order?
        order_cost = self.facility.get_config("order_cost")
        assert isinstance(order_cost, int)

        assert isinstance(self.data_model, ConsumerDataModel)
        self.data_model.initialize(order_cost=order_cost)

        if self.facility.upstreams is not None:
            # Construct sources from facility's upstreams.
            # List[FacilityBase]
            sources: list = self.facility.upstreams.get(self.product_id, [])

            if len(sources) > 0:  # TODO: update this part. the supplier can also request product from vendor.
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
                                    self.source_facility_id_list.append(source_facility.id)
                    else:
                        # If we are not a manufacturing, just check if upstream have this sku configuration.
                        if sku.id in source_facility.skus:
                            self.source_facility_id_list.append(source_facility.id)

    def _step_impl(self, tick: int):
        self._update_pending_order()

        if self.action is None:
            return

        assert isinstance(self.action, ConsumerAction)

        # NOTE: id == 0 means invalid,as our id is 1 based.
        if self.action.quantity <= 0 or self.action.product_id <= 0 or self.action.source_id == 0:
            return

        # NOTE: we are using product unit as destination,
        # so we expect the action.source_id is an id of product unit.
        self.update_open_orders(self.action.source_id, self.action.product_id, self.action.quantity)

        order = Order(self.facility, self.action.product_id, self.action.quantity, self.action.vlt)

        source_facility = self.world.get_facility_by_id(self.action.source_id)

        # Here the order cost is calculated by the upper distribution unit, with the sku price in that facility.
        self._order_product_cost = source_facility.distribution.place_order(order)
        # TODO: the order would be cancelled if there is no available vehicles, but the cost is not decreased
        # at that time.

        self._purchased = self.action.quantity

    def flush_states(self):
        if self._received > 0:
            self.data_model.received = self._received

        if self._purchased > 0:
            self.data_model.purchased = self._purchased
            self.data_model.latest_consumptions = 1.0

        if self._order_product_cost > 0:
            self.data_model.order_product_cost = self._order_product_cost

    def post_step(self, tick: int):
        if self._received > 0:
            self.data_model.received = 0
            self._received = 0

        if self._purchased > 0:
            self.data_model.purchased = 0
            self.data_model.latest_consumptions = 0
            self._purchased = 0

        if self._order_product_cost > 0:
            self.data_model.order_product_cost = 0
            self._order_product_cost = 0

    def reset(self):
        super(ConsumerUnit, self).reset()

        self._open_orders.clear()

        # Reset status in Python side.
        self._received = 0
        self._purchased = 0
        self._order_product_cost = 0
        self.source_facility_id_list = []
        self.pending_order_daily = [0] * self.world.configs.settings["pending_order_len"]

    def get_in_transit_quantity(self):
        quantity = 0

        for _, orders in self._open_orders.items():
            quantity += orders.get(self.product_id, 0)

        return quantity

    def _update_pending_order(self):
        self.pending_order_daily = shift(self.pending_order_daily, -1, cval=0)

    def get_unit_info(self) -> ConsumerUnitInfo:
        return ConsumerUnitInfo(
            **super(ConsumerUnit, self).get_unit_info().__dict__,
            source_facility_id_list=self.source_facility_id_list,
        )
