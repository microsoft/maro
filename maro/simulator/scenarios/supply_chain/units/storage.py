# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

from .unitbase import UnitBase


class StorageUnit(UnitBase):
    """Unit that used to store skus."""

    def __init__(self):
        super().__init__()

        # Used to map from product id to slot index.
        self.capacity = 0
        self.remaining_space = 0
        self.product_level = {}

        # Which product's number has changed.
        self._changed_product_cache = {}

    def try_add_products(self, product_quantities: Dict[int, int], all_or_nothing=True) -> dict:
        """Try to add products into storage.

        Args:
            product_quantities (Dict[int, int]): Dictionary of product id and quantity need to add to storage.
            all_or_nothing (bool): Failed if all product cannot be added, or add as many as it can. Default is True.

        Returns:
            dict: Dictionary of product id and quantity success added.
        """
        if all_or_nothing and self.remaining_space < sum(product_quantities.values()):
            return {}

        unloaded_quantities = {}

        for product_id, quantity in product_quantities.items():
            unload_quantity = min(self.remaining_space, quantity)

            self.product_level[product_id] += unload_quantity
            unloaded_quantities[product_id] = unload_quantity

            self._changed_product_cache[product_id] = True

            self.remaining_space -= unload_quantity

        return unloaded_quantities

    def try_take_products(self, product_quantities: Dict[int, int]) -> bool:
        """Try to take specified number of product.

        Args:
            product_quantities (Dict[int, int]): Dictionary of product id and quantity to take from storage.

        Returns:
            bool: Is success to take?
        """
        # Check if we can take all kinds of products?
        for product_id, quantity in product_quantities.items():
            if self.product_level[product_id] < quantity:
                return False

        # Take from storage.
        for product_id, quantity in product_quantities.items():
            self.product_level[product_id] -= quantity
            self._changed_product_cache[product_id] = True

            self.remaining_space += quantity

        return True

    def take_available(self, product_id: int, quantity: int) -> int:
        """Take as much as available specified product from storage.

        Args:
            product_id (int): Product to take.
            quantity (int): Max quantity to take.

        Returns:
            int: Actual quantity taken.
        """
        available = self.product_level[product_id]
        actual = min(available, quantity)

        self.product_level[product_id] -= actual
        self._changed_product_cache[product_id] = True

        self.remaining_space += actual

        return actual

    def get_product_number(self, product_id: int) -> int:
        """Get product number in storage.

        Args:
            product_id (int): Product to check.

        Returns:
            int: Available number of product.
        """
        return self.product_level[product_id]

    def initialize(self):
        super(StorageUnit, self).initialize()

        self.capacity = self.config.get("capacity", 100)
        self.remaining_space = self.capacity

        for sku in self.facility.skus.values():
            self.product_level[sku.id] = sku.init_stock
            self._changed_product_cache[sku.id] = False

            self.remaining_space -= sku.init_stock

        self.data_model.initialize(
            capacity=self.capacity,
            remaining_space=self.remaining_space,
            product_list=[sku_id for sku_id in self.product_level.keys()],
            product_number=[n for n in self.product_level.values()]
        )

    def flush_states(self):
        # Write the changes to frame.
        i = 0
        has_changes = False
        for product_id, product_number in self.product_level.items():
            if self._changed_product_cache[product_id]:
                has_changes = True
                self._changed_product_cache[product_id] = False

                self.data_model.product_number[i] = product_number
            i += 1

        if has_changes:
            self.data_model.remaining_space = self.remaining_space

    def reset(self):
        super(StorageUnit, self).reset()

        self.remaining_space = self.capacity

        for sku in self.facility.skus.values():
            # self.product_level[sku.id] = sku.init_stock
            # self.remaining_space -= sku.init_stock
            # if hasattr(sku, "sale_gamma") and (sku.sale_gamma is not None):
            #     init_stock = random.randint(0, 5) * sku.sale_gamma
            #     sku.init_stock = init_stock
            self.product_level[sku.id] = sku.init_stock
            self.remaining_space -= sku.init_stock

    def get_unit_info(self):
        info = super().get_unit_info()

        info["product_list"] = [i for i in self.product_level.keys()]

        return info
