
from .base import UnitBase

from typing import Dict


class StorageUnit(UnitBase):
    """Unit that used to store skus.
    """
    def __init__(self):
        super().__init__()

        # used to map from product id to slot index
        self.product_index_mapping: Dict[int, int] = {}

    def initialize(self, configs):
        super().initialize(configs)

        for index, product_id in enumerate(self.data.product_list[:]):
            self.product_index_mapping[product_id] = index

    def try_add_units(self, product_quantities: Dict[int, int], all_or_nothing=True) -> dict:
        if all_or_nothing and self.data.remaining_space < sum(product_quantities.values()):
            return {}

        unloaded_quantities = {}

        for product_id, quantity in product_quantities.items():
            unload_quantity = min(self.data.remaining_space, quantity)

            product_index = self.product_index_mapping[product_id]
            self.data.product_number[product_index] += unload_quantity
            unloaded_quantities[product_id] = unload_quantity

        return unloaded_quantities

    def try_take_units(self, product_quantities: Dict[int, int]):
        for product_id, quantity in product_quantities.items():
            product_index = self.product_index_mapping[product_id]

            if self.data.product_number[product_index] < quantity:
                return False

        # TODO: refactoring for dup code
        for product_id, quantity in product_quantities.items():
            product_index = self.product_index_mapping[product_id]

            self.data.product_number[product_index] -= quantity

        return True

    def take_available(self, product_id: int, quantity: int):
        product_index = self.product_index_mapping[product_id]
        available = self.data.product_number[product_index]
        actual = min(available, quantity)

        self.data.product_number[product_index] -= actual

        return actual

    def get_product_number(self, product_id: int) -> int:
        product_index = self.product_index_mapping[product_id]

        return self.data.product_number[product_index]
