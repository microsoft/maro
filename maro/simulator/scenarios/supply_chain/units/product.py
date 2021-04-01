# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from .balancesheet import BalanceSheet
from .consumer import ConsumerUnit
from .manufacture import ManufactureUnit
from .seller import SellerUnit
from .skuunit import SkuUnit
from .storage import StorageUnit
from .distribution import DistributionUnit

class ProductUnit(SkuUnit):
    """Unit that used to group units of one special sku, usually contains consumer, seller and manufacture."""

    # Consumer unit of current sku.
    consumer: ConsumerUnit = None

    # Seller unit of current sku.
    seller: SellerUnit = None

    # Manufacture unit of this sku.
    manufacture: ManufactureUnit = None

    # Storage of this facility, always a reference of facility.storage.
    storage: StorageUnit = None

    distribution: DistributionUnit = None

    total_step_balance: BalanceSheet = None

    def __init__(self):
        super().__init__()
        self.total_step_balance = BalanceSheet()

    def initialize(self):
        super(ProductUnit, self).initialize()

    def step(self, tick: int):
        for unit in self.children:
            unit.step(tick)

        self.deposit()

    def deposit(self):
        balance_sheets = []
        rewards = []

        for unit in self.children:
            balance_sheets.append(unit.step_balance_sheet)
            rewards.append(unit.step_reward)

        storage_reward = 0
        if self.storage is not None:
            storage_reward = - self.storage.product_number[self.storage.product_index_mapping[self.product_id]] * self.storage.unit_storage_cost

        storage_balance = BalanceSheet(0, storage_reward)

        balance_sheets.append(storage_balance)
        rewards.append(storage_reward)

        distribution_balance = BalanceSheet()

        if self.distribution is not None:
            check_order = self.distribution.check_in_order[self.product_id]
            transport_cost = self.distribution.transportation_cost[self.product_id]
            delay_order_penalty = self.distribution.delay_order_penalty[self.product_id]
            distribution_cost = -(transport_cost + delay_order_penalty)

            self.distribution.transportation_cost[self.product_id] = 0
            self.distribution.delay_order_penalty[self.product_id] = 0

            distribution_balance.profit = check_order * self.get_selling_price()
            distribution_balance.loss = distribution_cost

        balance_sheets.append(distribution_balance)
        rewards.append(distribution_balance.total())

        if self.product_id in self.facility.downstreams:
            for facility in self.facility.downstreams[self.product_id]:
                downstream_product = facility.products[self.product_id]
                balance_sheets.append(downstream_product.step_balance_sheet)
                rewards.append(downstream_product.step_reward)

        self.step_balance_sheet = sum(balance_sheets)

        self.step_reward = sum(rewards)

        self.total_step_balance += self.step_balance_sheet

    def flush_states(self):
        for unit in self.children:
            unit.flush_states()

    def post_step(self, tick: int):
        super(ProductUnit, self).post_step(tick)

        for unit in self.children:
            unit.post_step(tick)

    def reset(self):
        super(ProductUnit, self).reset()

        for unit in self.children:
            unit.reset()

    def get_unit_info(self) -> dict:
        return {
            "id": self.id,
            "sku_id": self.product_id,
            "max_vlt": self._get_max_vlt(),
            "node_name": type(self.data_model).__node_name__ if self.data_model is not None else None,
            "node_index": self.data_model_index if self.data_model is not None else None,
            "class": type(self),
            "config": self.config,
            "consumer": self.consumer.get_unit_info() if self.consumer is not None else None,
            "seller": self.seller.get_unit_info() if self.seller is not None else None,
            "manufacture": self.manufacture.get_unit_info() if self.manufacture is not None else None
        }

    def get_latest_sale(self):
        sale = 0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            sale += facility.products[self.product_id].get_latest_sale()

        return sale

    def get_sale_mean(self):
        sale_mean = 0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            sale_mean += facility.products[self.product_id].get_sale_mean()

        return sale_mean

    def get_sale_std(self):
        sale_std = 0

        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            sale_std += facility.products[self.product_id].get_sale_std()

        return sale_std / np.sqrt(max(1, len(downstreams)))

    def get_selling_price(self):
        price = 0.0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            price = max(price, facility.products[self.product_id].get_selling_price())

        return price

    def _get_max_vlt(self):
        vlt = 1

        if self.consumer is not None:
            for source_facility_id in self.consumer.sources:
                source_facility = self.world.get_facility_by_id(source_facility_id)

                source_vlt = source_facility.skus[self.product_id].vlt

                vlt = max(vlt, source_vlt) 

        return vlt

    @staticmethod
    def generate(facility, config: dict, unit_def):
        """Generate product unit by sku information.

        Args:
            facility (FacilityBase): Facility this product belongs to.
            config (dict): Config of children unit.
        """
        instance_list = {}

        if facility.skus is not None and len(facility.skus) > 0:
            world = facility.world

            for sku_id, sku in facility.skus.items():
                sku_type = getattr(sku, "type", None)

                product_unit: ProductUnit = world.build_unit_by_type(ProductUnit, facility, facility, unit_def)
                product_unit.product_id = sku_id
                product_unit.children = []
                product_unit.parse_configs(config)
                product_unit.storage = product_unit.facility.storage
                product_unit.distribution = product_unit.facility.distribution

                for child_name in ("manufacture", "consumer", "seller"):
                    conf = config.get(child_name, None)

                    if conf is not None:
                        # Ignore manufacture unit if it is not for a production, even it is configured in config.
                        if sku_type != "production" and child_name == "manufacture":
                            continue

                        if sku_type == "production" and child_name == "consumer":
                            continue

                        child_unit = world.build_unit(facility, product_unit, conf)
                        child_unit.product_id = sku_id

                        setattr(product_unit, child_name, child_unit)

                        # Parse config for unit.
                        child_unit.parse_configs(conf.get("config", {}))

                        product_unit.children.append(child_unit)

                instance_list[sku_id] = product_unit

        return instance_list
