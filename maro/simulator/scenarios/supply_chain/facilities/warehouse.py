# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List

from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit

from .facility import FacilityBase


class WarehouseFacility(FacilityBase):
    """Warehouse facility that used to storage products, composed with storage, distribution and product units."""

    SkuInfo = namedtuple("SkuInfo", ("name", "init_stock", "id", "price", "delay_order_penalty", "order_cost"))

    # Storage unit for this facility, must be a sub class of StorageUnit.
    storage: StorageUnit = None

    # Distribution unit for this facility.
    distribution: DistributionUnit = None

    # Product unit list for this facility.
    products: List[ProductUnit] = None

    def __init__(self):
        self.skus = {}

    def parse_skus(self, configs: dict):
        for sku_name, sku_config in configs.items():
            sku = self.world.get_sku_by_name(sku_name)

            sku_info = WarehouseFacility.SkuInfo(
                sku_name,
                sku_config["init_stock"],
                sku.id,
                sku_config.get("price", 0),
                sku_config.get("delay_order_penalty", 0),
                sku_config.get("order_cost", 0)
            )

            self.skus[sku.id] = sku_info
