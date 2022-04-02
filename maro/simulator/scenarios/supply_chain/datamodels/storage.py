# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("storage")
class StorageDataModel(DataModelBase):
    """Data model for storage unit."""
    capacity = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    remaining_space = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    unit_storage_cost = NodeAttribute(AttributeType.Float, 1, is_list=True)

    # original is , used to save product and its number
    product_list = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    product_quantity = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    product_storage_index = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    def __init__(self) -> None:
        super(StorageDataModel, self).__init__()

        self._capacity: Optional[List[int]] = None
        self._remaining_space: Optional[List[int]] = None
        self._unit_storage_cost: Optional[List[float]] = None
        self._product_list: Optional[List[int]] = None
        self._product_quantity: Optional[List[int]] = None
        self._product_storage_index: Optional[List[int]] = None

    def initialize(
        self,
        capacity: List[int],
        remaining_space: List[int],
        unit_storage_cost: List[float],
        product_list: List[int] = None,
        product_quantity: List[int] = None,
        product_storage_index: List[int] = None,
    ) -> None:
        self._capacity = capacity
        self._remaining_space = remaining_space
        self._unit_storage_cost = unit_storage_cost
        self._product_list = product_list
        self._product_quantity = product_quantity
        self._product_storage_index = product_storage_index

        self.reset()

    def reset(self) -> None:
        super(StorageDataModel, self).reset()

        for _capacity in self._capacity:
            self.capacity.append(_capacity)

        for _remaining_space in self._remaining_space:
            self.remaining_space.append(_remaining_space)

        for _cost in self._unit_storage_cost:
            self.unit_storage_cost.append(_cost)

        if self._product_list is not None:
            for id in self._product_list:
                self.product_list.append(id)

        if self._product_quantity is not None:
            for quantity in self._product_quantity:
                self.product_quantity.append(quantity)

        if self._product_storage_index is not None:
            for idx in self._product_storage_index:
                self.product_storage_index.append(idx)
