# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC


class FacilityBase(ABC):
    """Base of all facilities."""

    # Id of this facility.
    id: int = None

    # Name of this facility.
    name: str = None

    # World of this facility belongs to.
    world = None

    # Skus in this facility.
    skus: dict = None

    # Product units for each sku in this facility.
    # Key is sku(product) id, value is the instance of product unit.
    products: dict = None

    # Storage unit in this facility.
    storage = None

    # Distribution unit in this facility.
    distribution = None

    # Upstream facilities.
    # Key is sku id, value is the list of product unit from upstream.
    upstreams: dict = None

    # Configuration of this facility.
    configs: dict = None

    def parse_skus(self, configs: dict):
        """Parse sku information from config.

        Args:
            configs (dict): Configuration of skus belongs to this facility.
        """
        pass

    def parse_configs(self, configs: dict):
        """Parse configuration of this facility.

        Args:
            configs (dict): Configuration of this facility.
        """
        self.configs = configs

    def get_config(self, key: str, default: object = None):
        return default if self.configs is None else self.configs.get(key, default)

    def initialize(self):
        """Initialize this facility after frame is ready."""
        pass

    def step(self, tick: int):
        """Push facility to next step.

        Args:
            tick (int): Current simulator tick.
        """
        if self.storage is not None:
            self.storage.step(tick)

        if self.distribution is not None:
            self.distribution.step(tick)

        if self.products is not None:
            for product in self.products.values():
                product.step(tick)

    def flush_states(self):
        """Flush states into frame."""
        if self.storage is not None:
            self.storage.flush_states()

        if self.distribution is not None:
            self.distribution.flush_states()

        if self.products is not None:
            for product in self.products.values():
                product.flush_states()

    def post_step(self, tick: int):
        """Post processing at the end of step."""
        if self.storage is not None:
            self.storage.post_step(tick)

        if self.distribution is not None:
            self.distribution.post_step(tick)

        if self.products is not None:
            for product in self.products.values():
                product.post_step(tick)

    def reset(self):
        """Reset facility for new episode."""
        if self.storage is not None:
            self.storage.reset()

        if self.distribution is not None:
            self.distribution.reset()

        if self.products is not None:
            for product in self.products.values():
                product.reset()

    def get_node_info(self) -> dict:
        products_info = {}

        for product_id, product in self.products.items():
            products_info[product_id] = product.get_unit_info()

        return {
            "id": self.id,
            "name": self.name,
            "class": type(self),
            "units": {
                "storage": self.storage.get_unit_info() if self.storage is not None else None,
                "distribution": self.distribution.get_unit_info() if self.distribution is not None else None,
                "products": products_info
            }
        }
