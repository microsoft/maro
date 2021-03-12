
from .base import UnitBase

from typing import Dict


class StorageUnit(UnitBase):
    """Unit that used to store skus.
    """

    def __init__(self):
        super().__init__()

        # used to map from product id to slot index

        # we use these variables to hold changes at python side, flash to frame before taking snapshot
        self.product_number = []
        self.product_index_mapping: Dict[int, int] = {}
        self.capacity = 0
        self.remaining_space = 0

    def initialize(self, configs: dict, durations: int):
        super().initialize(configs, durations)

        self.capacity = self.data.capacity
        self.remaining_space = self.capacity

        for index, product_id in enumerate(self.data.product_list[:]):
            product_number = self.data.product_number[index]

            self.product_number.append(product_number)
            self.product_index_mapping[product_id] = index

            self.remaining_space -= product_number

    def try_add_units(self, product_quantities: Dict[int, int], all_or_nothing=True) -> dict:
        if all_or_nothing and self.remaining_space < sum(product_quantities.values()):
            return {}

        unloaded_quantities = {}

        for product_id, quantity in product_quantities.items():
            unload_quantity = min(self.data.remaining_space, quantity)

            product_index = self.product_index_mapping[product_id]
            self.product_number[product_index] += unload_quantity
            unloaded_quantities[product_id] = unload_quantity

        return unloaded_quantities

    def try_take_units(self, product_quantities: Dict[int, int]):
        for product_id, quantity in product_quantities.items():
            product_index = self.product_index_mapping[product_id]

            if self.product_number[product_index] < quantity:
                return False

        # TODO: refactoring for dup code
        for product_id, quantity in product_quantities.items():
            product_index = self.product_index_mapping[product_id]

            self.product_number[product_index] -= quantity

        return True

    def take_available(self, product_id: int, quantity: int):
        product_index = self.product_index_mapping[product_id]
        available = self.product_number[product_index]
        actual = min(available, quantity)

        self.product_number[product_index] -= actual

        return actual

    def get_product_number(self, product_id: int) -> int:
        product_index = self.product_index_mapping[product_id]

        return self.product_number[product_index]

    def begin_post_step(self, tick: int):
        # write the changes to frame
        for i, number in enumerate(self.product_number):
            self.data.product_number[i] = self.product_number[i]

        self.data.remaining_space = self.capacity - sum(self.product_number)

    def reset(self):
        super().reset()

        self.remaining_space = self.capacity

        # TODO: dup code
        for index, product_id in enumerate(self.data.product_list[:]):
            product_number = self.data.product_number[index]

            self.product_number.append(product_number)
            self.product_index_mapping[product_id] = index

            self.remaining_space -= product_number
