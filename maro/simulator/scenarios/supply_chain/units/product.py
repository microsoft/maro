# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
    max_vlt: Optional[int]


class ProductUnit(ExtendUnitBase):
    """Unit that used to group units of one specific SKU, usually contains consumer, seller and manufacture."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(ProductUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
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

        # 1st element: out product_id; 2nd element: self consumption / out product quantity
        self.bom_out_info_list: List[Tuple[int, float]] = []

        # Internal states to track distribution.
        self._check_in_quantity_in_order: int = 0
        self._transportation_cost: float = 0
        self._delay_order_penalty: float = 0

    def initialize(self) -> None:
        super().initialize()

        facility_sku = self.facility.skus[self.product_id]

        assert isinstance(self.data_model, ProductDataModel)
        self.data_model.initialize(facility_sku.price)

    def init_step(self, tick: int) -> None:
        for unit in self.children:
            unit.init_step(tick)

        if self._check_in_quantity_in_order > 0:
            self.data_model.check_in_quantity_in_order = 0
            self._check_in_quantity_in_order = 0

        if self._transportation_cost > 0:
            self.data_model.transportation_cost = 0
            self._transportation_cost = 0

        if self._delay_order_penalty > 0:
            self.data_model.delay_order_penalty = 0
            self._delay_order_penalty = 0

    def step(self, tick: int) -> None:
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

        if self._check_in_quantity_in_order > 0:
            self.data_model.check_in_quantity_in_order = self._check_in_quantity_in_order

        if self._transportation_cost > 0:
            self.data_model.transportation_cost = self._transportation_cost

        if self._delay_order_penalty > 0:
            self.data_model.delay_order_penalty = self._delay_order_penalty

    def post_step(self, tick: int) -> None:
        for unit in self.children:
            unit.post_step(tick)

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
            max_vlt=self.facility.get_max_vlt(self.product_id),
        )

    def _get_sale_means(self) -> List[float]:
        sale_means = []
        _cache: Dict[int, float] = {}

        def _get_sale_mean(product_unit: ProductUnit) -> float:
            if product_unit.id not in _cache:
                _cache[product_unit.id] = product_unit.get_sale_mean()
            return _cache[product_unit.id]

        for downstream_facility in self.facility.downstream_facility_list[self.product_id]:
            sale_means.append(_get_sale_mean(downstream_facility.products[self.product_id]))
        for out_product_id, consumption_ratio in self.bom_out_info_list:
            sale_means.append(int(_get_sale_mean(self.facility.products[out_product_id]) * consumption_ratio))

        return sale_means

    def get_sale_mean(self) -> float:
        """"Here the sale mean of upstreams means the sum of its downstreams,
        which indicates the daily demand of this product from the aspect of the facility it belongs."""
        return float(np.sum(self._get_sale_means()))

    def get_sale_std(self) -> float:
        sale_means = self._get_sale_means()
        return 0.0 if len(sale_means) == 0 else float(np.std(sale_means))

    def get_max_sale_price(self) -> float:
        price = 0.0

        for downstream_facility in self.facility.downstream_facility_list[self.product_id]:
            price = max(price, downstream_facility.products[self.product_id].get_max_sale_price())

        return price


class StoreProductUnit(ProductUnit):
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(StoreProductUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

    def get_sale_mean(self) -> float:
        return self.seller.sale_mean()

    def get_sale_std(self) -> float:
        return self.seller.sale_std()

    def get_max_sale_price(self) -> float:
        return self.facility.skus[self.product_id].price
