# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List

from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit

from .facility import FacilityBase


class SupplierFacility(FacilityBase):
    """Supplier facilities used to produce products with material products."""

    SkuInfo = namedtuple(
        "SkuInfo",
        (
            "name",
            "id",
            "init_stock",
            "type",
            "cost",
            "price",
            "delay_order_penalty",
            "product_unit_cost",
            "order_cost"
        )
    )

    # Storage unit of this facility.
    storage: StorageUnit

    # Distribution unit of this facility.
    distribution: DistributionUnit

    # Product unit list of this facility.
    products: List[ProductUnit]

    def __init__(self):
        self.skus = {}

    def parse_skus(self, configs: dict):
        for sku_name, sku_config in configs.items():
            sku = self.world.get_sku_by_name(sku_name)
            sku_info = SupplierFacility.SkuInfo(
                sku_name,
                sku.id,
                sku_config["init_stock"],
                sku_config["type"],
                sku_config.get("cost", 0),
                sku_config.get("price", 0),
                sku_config.get("delay_order_penalty", 0),
                sku_config.get("product_unit_cost", 1),
                sku_config.get("order_cost", 0)
            )

            self.skus[sku.id] = sku_info
