# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from .unitbase import UnitBase, BaseUnitInfo

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


DEFAULT_SUB_STORAGE_ID = 0


@dataclass
class SubStorageConfig:
    id: int
    capacity: int = 100  # TODO: Is it a MUST config or could it be default?
    unit_storage_cost: int = 1


def parse_storage_config(config: dict) -> List[SubStorageConfig]:  # TODO: here or in parser
    if not isinstance(config, list):
        id = config.get("id", DEFAULT_SUB_STORAGE_ID)
        return [SubStorageConfig(id=id, **config)]
    return [SubStorageConfig(**cfg) for cfg in config]


class AddStrategy(Enum):
    # Ignore the pre-set upper bound for each product, successfully added all products if renaming space is enough to
    # add all, else failed to add any product.
    IgnoreUpperBoundAllOrNothing = 1
    # Ignore the pre-set upper bound for each product, add coming products proportional to the coming quantity.
    IgnoreUpperBoundProportional = 2
    # Ignore the pre-set upper bound for each product, add coming products one by one in order (first come, first add).
    IgnoreUpperBoundAddInOrder = 3
    # Limited by the pre-set upper bound for each product. Assuming the sum(upper bound) == capacity, the success of
    # adding different products are independent with each other.
    LimitedByUpperBound = 4


@dataclass
class StorageUnitInfo(BaseUnitInfo):
    product_list: List[int]


class StorageUnit(UnitBase):
    """Unit that used to store skus."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(StorageUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

        # Key: Sub-Storage ID
        self._capacity_dict: Dict[int, int] = {}
        self._remaining_space_dict: Dict[int, int] = {}
        self._unit_cost_dict: Dict[int, float] = {}
        self._total_capacity: int = 0

        # 1st-key: the Sub-Storage ID, 2nd-key: the SKU ID, value: the upper bound.
        # None value indicates non-pre-limited.
        self._storage_sku_upper_bound: Dict[int, Dict[int, Optional[int]]] = defaultdict(dict)

        # Key: product_id
        self._product_level: Dict[int, int] = {}
        self._product_level_changed: Dict[int, bool] = {}

        # Mapping from the product id to sub-storage id.
        self._product2storage: Dict[int, int] = {}
        # Mapping between the sub-storage id and the idx in the data model.
        self._storage_id2idx: Dict[int, int] = {}
        self._storage_idx2id: Dict[int, int] = {}

    @property
    def capacity(self) -> int:  # TODO: not used now. Check to remove or not.
        return self._total_capacity

    @property
    def remaining_space(self) -> int:  # TODO: not used now. Check to remove or not.
        return sum(self._remaining_space_dict.values())

    def initialize(self) -> None:
        super(StorageUnit, self).initialize()

        # Initialize capacity info.
        self.config: List[SubStorageConfig] = parse_storage_config(self.config)
        for sub_config in self.config:
            assert sub_config.id not in self._capacity_dict, f"Sub-Storage {sub_config.id} already exist!"
            self._capacity_dict[sub_config.id] = sub_config.capacity
            self._remaining_space_dict[sub_config.id] = sub_config.capacity
            self._unit_cost_dict[sub_config.id] = sub_config.unit_storage_cost
            self._total_capacity += sub_config.capacity

        # Initialize the product stock level
        for sku in self.facility.skus.values():
            self._product_level[sku.id] = sku.init_stock
            self._product_level_changed[sku.id] = False

            self._product2storage[sku.id] = sku.sub_storage_id
            assert sku.sub_storage_id in self._remaining_space_dict
            self._remaining_space_dict[sku.sub_storage_id] -= sku.init_stock
            assert self._remaining_space_dict[sku.sub_storage_id] >= 0, (
                f"Initial stock too much for Sub Storage {sku.sub_storage_id} of Facility {self.facility.name}!"
            )
            self._storage_sku_upper_bound[sku.sub_storage_id][sku.id] = sku.storage_upper_bound

        # Initialize the None upper bound SKU with the average remaining space.
        for sub_storage_id, upper_bound_dict in self._storage_sku_upper_bound.items():
            # Count how many sku has None upper bound and the remaining space.
            remaining_space = self._capacity_dict[sub_storage_id]
            sku_num = 0
            for upper_bound in upper_bound_dict.values():
                if upper_bound:
                    remaining_space -= upper_bound
                else:
                    sku_num += 1

            if sku_num == 0 and remaining_space > 0:
                # TODO: decide to evenly expand the upper bound or not.
                print(
                    f"The given upper bound cannot fill the whole capacity of Sub storage {sub_storage_id} "
                    f"in facility {self.facility.name}"
                )

            # TODO: Can Sum(Upper Bound) > Capacity?

            # Divide the remaining space evenly among the SKUs with None upper bound.
            average_space = remaining_space // sku_num
            for sku_id, upper_bound in upper_bound_dict.items():
                if upper_bound is None:
                    sku_num -= 1
                    if sku_num > 0:
                        upper_bound_dict[sku_id] = average_space
                        remaining_space -= average_space
                    else:
                        # In the case the initial remaining space is not divisible.
                        upper_bound_dict[sku_id] = remaining_space

        capacity_list: List[int] = []
        remaining_space_list: List[int] = []
        unit_cost_list: List[float] = []
        for idx, id in enumerate(self._capacity_dict.keys()):
            self._storage_id2idx[id] = idx
            self._storage_idx2id[idx] = id

            capacity_list.append(self._capacity_dict[id])
            remaining_space_list.append(self._remaining_space_dict[id])
            unit_cost_list.append(self._unit_cost_dict[id])

        self.data_model.initialize(
            capacity=capacity_list,
            remaining_space=remaining_space_list,
            unit_storage_cost=unit_cost_list,
            product_list=[sku_id for sku_id in self._product_level.keys()],
            product_quantity=[n for n in self._product_level.values()],
            product_storage_index=[
                self._storage_id2idx[self._product2storage[sku_id]] for sku_id in self._product_level.keys()
            ],
        )

    def get_product_quantity(self, product_id: int) -> int:
        """Get product quantity in storage.

        Args:
            product_id (int): Product to check.

        Returns:
            int: Available quantity of product.
        """
        return self._product_level[product_id]

    def get_product_max_remaining_space(self, product_id: int) -> int:
        # ! Currently the upper bound is only used in Manufacture Unit.
        remaining_space = self._remaining_space_dict[self._product2storage[product_id]]

        upper_bound = self._storage_sku_upper_bound[self._product2storage[product_id]][product_id]
        remaining_quota = max(upper_bound - self._product_level[product_id], 0)

        return min(remaining_space, remaining_quota)

    def _add_product(self, product_id: int, quantity: int) -> None:
        assert self._remaining_space_dict[self._product2storage[product_id]] >= quantity
        self._product_level[product_id] += quantity
        self._product_level_changed[product_id] = True
        self._remaining_space_dict[self._product2storage[product_id]] -= quantity

    def _take_product(self, product_id: int, quantity: int) -> None:
        assert self._product_level[product_id] >= quantity
        self._product_level[product_id] -= quantity
        self._product_level_changed[product_id] = True
        self._remaining_space_dict[self._product2storage[product_id]] += quantity

    def try_add_products(
        self,
        product_quantities: Dict[int, int],
        add_strategy: AddStrategy = AddStrategy.IgnoreUpperBoundAllOrNothing,
    ) -> Dict[int, int]:
        """Try to add products into storage.

        Args:
            product_quantities (Dict[int, int]): Dictionary of product id and quantity need to add to storage.
            add_strategy (AddStrategy): The strategy to add products into the storage, Defaults to
                AddStrategy.IgnoreUpperBoundAddAllOrNothing.

        Returns:
            Dict[int, int]: Dictionary of product id and quantity success added.
        """

        added_quantities: Dict[int, int] = {}

        if add_strategy in [AddStrategy.IgnoreUpperBoundAllOrNothing, AddStrategy.IgnoreUpperBoundProportional]:
            space_requirements: Dict[int, int] = defaultdict(lambda: 0)
            for product_id, quantity in product_quantities.items():
                storage_id = self._product2storage[product_id]
                space_requirements[storage_id] += quantity

            if add_strategy == AddStrategy.IgnoreUpperBoundAllOrNothing:
                for storage_id, requirement in space_requirements.items():
                    if self._remaining_space_dict[storage_id] < requirement:
                        return {}

                for product_id, quantity in product_quantities.items():
                    self._add_product(product_id, quantity)
                    added_quantities[product_id] = quantity

            elif add_strategy == AddStrategy.IgnoreUpperBoundProportional:
                fulfill_ratio_dict: Dict[int, float] = {}
                for storage_id, requirement in space_requirements.items():
                    fulfill_ratio_dict[storage_id] = min(1.0, self._remaining_space_dict[storage_id] / requirement)

                for product_id, quantity in product_quantities.items():
                    storage_id = self._product2storage[product_id]
                    quantity = min(
                        int(quantity * fulfill_ratio_dict[storage_id]), self._remaining_space_dict[storage_id],
                    )
                    self._add_product(product_id, quantity)
                    added_quantities[product_id] = quantity

        elif add_strategy == AddStrategy.IgnoreUpperBoundAddInOrder:
            for product_id, quantity in product_quantities.items():
                quantity = min(quantity, self._remaining_space_dict[self._product2storage[product_id]])
                self._add_product(product_id, quantity)
                added_quantities[product_id] = quantity

        elif add_strategy == AddStrategy.LimitedByUpperBound:
            for product_id, quantity in product_quantities.items():
                quantity = min(quantity, self.get_product_max_remaining_space(product_id))
                self._add_product(product_id, quantity)
                added_quantities[product_id] = quantity

        else:
            raise ValueError(f"Unrecognized storage add strategy: {add_strategy}!")

        return added_quantities

    def try_take_products(self, product_quantities: Dict[int, int]) -> bool:
        """Try to take specified number of product.

        Args:
            product_quantities (Dict[int, int]): Dictionary of product id and quantity to take from storage.

        Returns:
            bool: Is success to take?
        """
        # Check if we can take all kinds of products?
        for product_id, quantity in product_quantities.items():
            if self._product_level[product_id] < quantity:
                return False

        # Take from storage.
        for product_id, quantity in product_quantities.items():
            self._take_product(product_id, quantity)

        return True

    def take_available(self, product_id: int, quantity: int) -> int:
        """Take as much as available specified product from storage.

        Args:
            product_id (int): Product to take.
            quantity (int): Max quantity to take.

        Returns:
            int: Actual quantity taken.
        """
        available = self._product_level[product_id]
        actual = min(available, quantity)
        self._take_product(product_id, actual)
        return actual

    def flush_states(self) -> None:
        # Write the changes to frame.
        i = 0
        has_changes = False
        for product_id, quantity in self._product_level.items():
            if self._product_level_changed[product_id]:
                has_changes = True
                self._product_level_changed[product_id] = False

                self.data_model.product_quantity[i] = quantity
            i += 1

        if has_changes:
            i = 0
            for remaining_space in self._remaining_space_dict.values():
                self.data_model.remaining_space[i] = remaining_space
                i += 1

    def reset(self) -> None:
        super(StorageUnit, self).reset()

        # Reset status in Python side.
        for sub_config in self.config:
            self._remaining_space_dict[sub_config.id] = sub_config.capacity

        for sku in self.facility.skus.values():
            self._product_level[sku.id] = sku.init_stock
            self._product_level_changed[sku.id] = False

            self._remaining_space_dict[sku.sub_storage_id] -= sku.init_stock

    def get_unit_info(self) -> StorageUnitInfo:
        return StorageUnitInfo(
            **super(StorageUnit, self).get_unit_info().__dict__,
            product_list=[i for i in self._product_level.keys()],
        )
