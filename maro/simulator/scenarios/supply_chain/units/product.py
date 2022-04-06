# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from maro.simulator.scenarios.supply_chain.datamodels import ProductDataModel

from .consumer import ConsumerUnit, ConsumerUnitInfo
from .distribution import DistributionUnit
from .extendunitbase import ExtendUnitBase, ExtendUnitInfo
from .manufacture import ManufactureUnit, ManufactureUnitInfo
from .seller import SellerUnit, SellerUnitInfo
from .storage import StorageUnit


@dataclass
class ProductUnitInfo(ExtendUnitInfo):
    consumer_info: Optional[ConsumerUnitInfo]
    manufacture_info: Optional[ManufactureUnitInfo]
    seller_info: Optional[SellerUnitInfo]
    max_vlt: int


class ProductUnit(ExtendUnitBase):
    """Unit that used to group units of one specific SKU, usually contains consumer, seller and manufacture."""

    def __init__(self) -> None:
        super(ProductUnit, self).__init__()

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

    @staticmethod
    def generate(facility, config: dict, unit_def: object) -> Dict[int, ProductUnit]:
        """Generate product unit by sku information.

        Args:
            facility (FacilityBase): Facility this product belongs to.
            config (dict): Config of children unit.
            unit_def (object): Definition of the unit (from config).

        Returns:
            dict: Dictionary of product unit, key is the product id, value is ProductUnit.
        """
        products_dict: Dict[int, ProductUnit] = {}

        if facility.skus is not None and len(facility.skus) > 0:
            world = facility.world

            for sku_id, sku in facility.skus.items():
                sku_type = sku.type

                product_unit: ProductUnit = world.build_unit_by_type(unit_def, facility, facility)
                product_unit.product_id = sku_id
                product_unit.children = []
                product_unit.parse_configs(config)
                product_unit.storage = product_unit.facility.storage
                product_unit.distribution = product_unit.facility.distribution

                # NOTE: BE CAREFUL about the order, product unit will use this order update children,
                # the order may affect the states.
                # Here we make sure consumer is the first one, so it can place order first.
                for child_name in ("consumer", "seller", "manufacture"):
                    conf = config.get(child_name, None)

                    if conf is not None:
                        # Ignore manufacture unit if it is not for a production, even it is configured in config.
                        if sku_type != "production" and child_name == "manufacture":
                            continue

                        # We produce the product, so we do not need to purchase it.
                        if sku_type == "production" and child_name == "consumer":
                            continue

                        child_unit = world.build_unit(facility, product_unit, conf)
                        child_unit.product_id = sku_id

                        setattr(product_unit, child_name, child_unit)

                        # Parse config for unit.
                        child_unit.parse_configs(conf.get("config", {}))

                        product_unit.children.append(child_unit)

                products_dict[sku_id] = product_unit

        return products_dict
