# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List

from maro.simulator.scenarios.supply_chain.units import ProductUnit, StorageUnit

from .facility import FacilityBase


class RetailerFacility(FacilityBase):
    """Retail facility used to generate order from upstream, and sell products by demand."""

    SkuInfo = namedtuple(
        "SkuInfo",
        (
            "name",
            "id",
            "price",
            "cost",
            "init_stock",
            "sale_gamma",
            "order_cost",
            "backlog_ratio"
        )
    )

    # Product unit list of this facility.
    products: List[ProductUnit]

    # Storage unit of this facility.
    storage: StorageUnit

    def __init__(self):
        self.skus = {}

    def parse_skus(self, configs: dict):
        for sku_name, sku_config in configs.items():
            sku = self.world.get_sku_by_name(sku_name)
            sku_info = RetailerFacility.SkuInfo(
                sku_name,
                sku.id,
                sku_config.get("price", 0),
                sku_config.get("cost", 0),
                sku_config["init_stock"],
                sku_config.get("sale_gamma", 0),
                sku_config.get("order_cost", 0),
                sku_config.get("backlog_ratio", 0)
            )

            self.skus[sku.id] = sku_info
