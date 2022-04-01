# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

from maro.simulator.scenarios.supply_chain.datamodels import ProductDataModel

from .consumer import ConsumerUnit, ConsumerUnitInfo
from .distribution import DistributionUnit
from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .manufacture import ManufactureUnit, ManufactureUnitInfo
from .seller import SellerUnit, SellerUnitInfo
from .storage import StorageUnit
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class ProductUnitInfo(ExtendUnitInfo):
    consumer_info: Optional[ConsumerUnitInfo]
    manufacture_info: Optional[ManufactureUnitInfo]
    seller_info: Optional[SellerUnitInfo]
    max_vlt: int


class ProductUnit(ExtendUnitBase):
    """Unit that used to group units of one specific SKU, usually contains consumer, seller and manufacture."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(ProductUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

        # The consumer unit of this SKU.
        self.consumer: Optional[ConsumerUnit] = None
        # The seller unit of this SKU.
        self.seller: Optional[SellerUnit] = None
        # The manufacture unit of this SKU.
        self.manufacture: Optional[ManufactureUnit] = None

        # The storage unit of the facility it belongs to. It is a reference to self.facility.storage.
        self.storage: Optional[StorageUnit] = None
        # The distribution unit of the facility it belongs to. It is a reference to self.facility.distribution.
        self.distribution: Optional[DistributionUnit] = None

        # Internal states to track distribution.
        self._check_in_quantity_in_order: int = 0
        self._transportation_cost: float = 0
        self._delay_order_penalty: float = 0

    def initialize(self) -> None:
        super().initialize()

        facility_sku = self.facility.skus[self.product_id]

        assert isinstance(self.data_model, ProductDataModel)
        self.data_model.initialize(facility_sku.price)

    def _step_impl(self, tick: int) -> None:
        for unit in self.children:
            unit.step(tick)

    def flush_states(self) -> None:
        for unit in self.children:
            unit.flush_states()

        if self.distribution is not None:
            # Processing in flush_states() to make sure self.distribution.step() has already done.
            self._check_in_quantity_in_order = self.distribution.check_in_quantity_in_order[self.product_id]
            self._transportation_cost = self.distribution.transportation_cost[self.product_id]
            self._delay_order_penalty = self.distribution.delay_order_penalty[self.product_id]

            self.distribution.check_in_quantity_in_order[self.product_id] = 0
            self.distribution.transportation_cost[self.product_id] = 0
            self.distribution.delay_order_penalty[self.product_id] = 0

        if self._check_in_quantity_in_order > 0:
            self.data_model.check_in_quantity_in_order = self._check_in_quantity_in_order

        if self._transportation_cost > 0:
            self.data_model.transportation_cost = self._transportation_cost

        if self._delay_order_penalty > 0:
            self.data_model.delay_order_penalty = self._delay_order_penalty

    def post_step(self, tick: int) -> None:
        super().post_step(tick)

        for unit in self.children:
            unit.post_step(tick)

        if self._check_in_quantity_in_order > 0:
            self.data_model.check_in_quantity_in_order = 0
            self._check_in_quantity_in_order = 0

        if self._transportation_cost > 0:
            self.data_model.transportation_cost = 0
            self._transportation_cost = 0

        if self._delay_order_penalty > 0:
            self.data_model.delay_order_penalty = 0
            self._delay_order_penalty = 0

    def reset(self) -> None:
        super().reset()

        self._check_in_quantity_in_order = 0
        self._transportation_cost = 0
        self._delay_order_penalty = 0

        for unit in self.children:
            unit.reset()

    def get_unit_info(self) -> ProductUnitInfo:
        return ProductUnitInfo(
            **super(ProductUnit, self).get_unit_info().__dict__,
            consumer_info=self.consumer.get_unit_info() if self.consumer else None,
            manufacture_info=self.manufacture.get_unit_info() if self.manufacture else None,
            seller_info=self.seller.get_node_info() if self.seller else None,
            max_vlt=self._get_max_vlt(),
        )

    def get_sale_mean(self) -> float:
        """"Here the sale mean of up-streams means the sum of its down-streams,
        which indicates the daily demand of this product from the aspect of the facility it belongs."""
        sale_mean = 0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            sale_mean += facility.products[self.product_id].get_sale_mean()

        return sale_mean

    def get_sale_std(self) -> float:
        sale_std = 0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            sale_std += facility.products[self.product_id].get_sale_std()

        return sale_std / np.sqrt(max(1, len(downstreams)))

    def get_max_sale_price(self) -> float:
        price = 0.0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            price = max(price, facility.products[self.product_id].get_max_sale_price())

        return price

    def _get_max_vlt(self) -> int:
        # TODO: update with vlt logic
        vlt = 1

        if self.consumer is not None:
            for f_id in self.consumer.source_facility_id_list:
                vlt = max(vlt, self.world.get_facility_by_id(f_id).skus[self.product_id].vlt)

        return vlt
