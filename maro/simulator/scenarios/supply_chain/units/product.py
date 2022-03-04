# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from ..datamodels import ProductDataModel
from .consumer import ConsumerUnit
from .distribution import DistributionUnit
from .extendunitbase import ExtendUnitBase
from .manufacture import ManufactureUnit
from .seller import SellerUnit
from .storage import StorageUnit


class ProductUnit(ExtendUnitBase):
    """Unit that used to group units of one special sku, usually contains consumer, seller and manufacture."""

    # Consumer unit of current sku.
    consumer: ConsumerUnit = None

    # Seller unit of current sku.
    seller: SellerUnit = None

    # Manufacture unit of this sku.
    manufacture: ManufactureUnit = None

    # Storage of this facility, always a reference of facility.storage.
    storage: StorageUnit = None

    # Reference to facility's distribution unit.
    distribution: DistributionUnit = None

    # Internal states to track distribution.
    _checkin_order = 0
    _transport_cost = 0
    _delay_order_penalty = 0

    def initialize(self) -> None:
        super().initialize()

        facility_sku = self.facility.skus[self.product_id]

        assert isinstance(self.data_model, ProductDataModel)
        self.data_model.initialize(facility_sku.price)

    def step(self, tick: int) -> None:
        for unit in self.children:
            unit.step(tick)

    def flush_states(self) -> None:
        for unit in self.children:
            unit.flush_states()

        if self.distribution is not None:
            self._checkin_order = self.distribution.check_in_order[self.product_id]
            self._transport_cost = self.distribution.transportation_cost[self.product_id]
            self._delay_order_penalty = self.distribution.delay_order_penalty[self.product_id]

            self.distribution.check_in_order[self.product_id] = 0
            self.distribution.transportation_cost[self.product_id] = 0
            self.distribution.delay_order_penalty[self.product_id] = 0

        if self._checkin_order > 0:
            self.data_model.distribution_check_order = self._checkin_order

        if self._transport_cost > 0:
            self.data_model.distribution_transport_cost = self._transport_cost

        if self._delay_order_penalty > 0:
            self.data_model.distribution_delay_order_penalty = self._delay_order_penalty

    def post_step(self, tick: int) -> None:
        super().post_step(tick)

        for unit in self.children:
            unit.post_step(tick)

        if self._checkin_order > 0:
            self.data_model.distribution_check_order = 0
            self._checkin_order = 0

        if self._transport_cost > 0:
            self.data_model.distribution_transport_cost = 0
            self._transport_cost = 0

        if self._delay_order_penalty > 0:
            self.data_model.distribution_delay_order_penalty = 0
            self._delay_order_penalty = 0

    def reset(self) -> None:
        super().reset()

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
            "manufacture": self.manufacture.get_unit_info() if self.manufacture is not None else None,
        }

    # TODO: add following field into states.
    def get_latest_sale(self) -> float:
        sale = 0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            sale += facility.products[self.product_id].get_latest_sale()

        return sale

    def get_sale_mean(self) -> float:
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

    def get_selling_price(self) -> float:
        price = 0.0
        downstreams = self.facility.downstreams.get(self.product_id, [])

        for facility in downstreams:
            price = max(price, facility.products[self.product_id].get_selling_price())

        return price

    def _get_max_vlt(self) -> int:
        vlt = 1

        if self.consumer is not None:
            for source_facility_id in self.consumer.sources:
                source_facility = self.world.get_facility_by_id(source_facility_id)

                source_vlt = source_facility.skus[self.product_id].vlt

                vlt = max(vlt, source_vlt)

        return vlt

    @staticmethod
    def generate(facility, config: dict, unit_def: object) -> dict:
        """Generate product unit by sku information.

        Args:
            facility (FacilityBase): Facility this product belongs to.
            config (dict): Config of children unit.
            unit_def (object): Definition of the unit (from config).

        Returns:
            dict: Dictionary of product unit, key is the product id, value if ProductUnit.
        """
        products_dict = {}

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
