# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .consumer import ConsumerUnit
from .manufacture import ManufactureUnit
from .seller import SellerUnit
from .skuunit import SkuUnit
from .storage import StorageUnit


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

    def step(self, tick: int):
        if self.consumer is not None:
            self.consumer.step(tick)

        if self.manufacture is not None:
            self.manufacture.step(tick)

        if self.seller is not None:
            self.seller.step(tick)

    def flush_states(self):
        if self.consumer is not None:
            self.consumer.flush_states()

        if self.manufacture is not None:
            self.manufacture.flush_states()

        if self.seller is not None:
            self.seller.flush_states()

    def post_step(self, tick: int):
        super(ProductUnit, self).post_step(tick)

        if self.consumer is not None:
            self.consumer.post_step(tick)

        if self.manufacture is not None:
            self.manufacture.post_step(tick)

        if self.seller is not None:
            self.seller.post_step(tick)

    def reset(self):
        super(ProductUnit, self).reset()

        if self.consumer is not None:
            self.consumer.reset()

        if self.manufacture is not None:
            self.manufacture.reset()

        if self.seller is not None:
            self.seller.reset()

    def get_unit_info(self) -> dict:
        return {
            "id": self.id,
            "sku_id": self.product_id,
            "node_name": type(self.data_model).__node_name__ if self.data_model is not None else None,
            "node_index": self.data_model_index if self.data_model is not None else None,
            "class": type(self),
            "consumer": self.consumer.get_unit_info() if self.consumer is not None else None,
            "seller": self.seller.get_unit_info() if self.seller is not None else None,
            "manufacture": self.manufacture.get_unit_info() if self.manufacture is not None else None
        }

    @staticmethod
    def generate(facility, config: dict):
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

                product_unit: ProductUnit = world.build_unit_by_type(ProductUnit, facility, facility)
                product_unit.product_id = sku_id

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
                        child_unit.parse_configs(conf)

                instance_list[sku_id] = product_unit

        return instance_list
