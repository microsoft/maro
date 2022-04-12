# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Union

from scipy.ndimage.interpolation import shift

from maro.simulator.scenarios.supply_chain.actions import ConsumerAction
from maro.simulator.scenarios.supply_chain.datamodels import ConsumerDataModel
from maro.simulator.scenarios.supply_chain.order import Order

from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World



@dataclass
class ConsumerUnitInfo(ExtendUnitInfo):
    source_facility_id_list: List[int]


class ConsumerUnit(ExtendUnitBase):
    """Consumer unit used to generate order to purchase from upstream by action."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(ConsumerUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

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

        assert isinstance(self.data_model, ConsumerDataModel)

        unit_order_cost = self.facility.skus[self.product_id].unit_order_cost
        self.data_model.initialize(order_cost=unit_order_cost)  # TODO: rename to unit_order_cost

        self.source_facility_id_list = [
            info.src_facility.id for info in self.facility.upstream_vlt_infos.get(self.product_id, [])
        ] if self.facility.upstream_vlt_infos is not None else []

    def _step_impl(self, tick: int):
        self._update_pending_order()

        if self.action is None:
            return

        assert isinstance(self.action, ConsumerAction)

        # NOTE: id == 0 means invalid,as our id is 1 based.
        if self.action.quantity <= 0 or self.action.product_id <= 0 or self.action.source_id == 0:
            return

        assert self.action.source_id in self.source_facility_id_list
        assert self.action.product_id == self.product_id

        vlt: Optional[int] = None
        for upstream_info in self.facility.upstream_vlt_infos[self.product_id]:
            if (
                upstream_info.src_facility.id == self.action.source_id
                and upstream_info.vehicle_type == self.action.vehicle_type
            ):
                vlt = upstream_info.vlt
                break

        assert vlt is not None

        # NOTE: we are using product unit as destination,
        # so we expect the action.source_id is an id of product unit.
        self.update_open_orders(self.action.source_id, self.action.product_id, self.action.quantity)

        order = Order(
            destination=self.facility,
            product_id=self.product_id,
            quantity=self.action.quantity,
            vehicle_type=self.action.vehicle_type,
            vlt=vlt  # TODO: add random factor if needed
        )

        source_facility = self.world.get_facility_by_id(self.action.source_id)

        # Here the order cost is calculated by the upper distribution unit, with the sku price in that facility.
        self._order_product_cost = source_facility.distribution.place_order(order)
        # TODO: the order would be cancelled if there is no available vehicles, but the cost is not decreased at that time.

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
        self.pending_order_daily = [0] * self.world.configs.settings["pending_order_len"]

    def get_in_transit_quantity(self):
        quantity = 0

        for _, orders in self._open_orders.items():
            quantity += orders.get(self.product_id, 0)
        # assert quantity >= 0, "wrong open orders"
        return quantity

    def _update_pending_order(self):
        self.pending_order_daily = shift(self.pending_order_daily, -1, cval=0)

    def get_unit_info(self) -> ConsumerUnitInfo:
        return ConsumerUnitInfo(
            **super(ConsumerUnit, self).get_unit_info().__dict__,
            source_facility_id_list=self.source_facility_id_list,
        )
