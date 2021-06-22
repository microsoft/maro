# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("storage")
class StorageDataModel(DataModelBase):
    """Data model for storage unit."""
    remaining_space = NodeAttribute(AttributeType.UInt)
    capacity = NodeAttribute(AttributeType.UInt, is_const=True)

    # original is , used to save product and its number
    product_list = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    product_number = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    unit_storage_cost = NodeAttribute(AttributeType.Float, 1, is_const=True)

    def __init__(self):
        super(StorageDataModel, self).__init__()

        self._capacity = 0
        self._remaining_space = None
        self._product_list = None
        self._product_number = None
        self._unit_storage_cost = 0

    def initialize(
        self,
        unit_storage_cost: float = 0,
        capacity: int = 0,
        remaining_space: int = None,
        product_list: list = None,
        product_number: list = None
    ):
        self._capacity = capacity
        self._remaining_space = remaining_space
        self._product_list = product_list
        self._product_number = product_number
        self._unit_storage_cost = unit_storage_cost

        self.reset()

    def reset(self):
        super(StorageDataModel, self).reset()

        self.capacity = self._capacity
        self.unit_storage_cost = self._unit_storage_cost

        if self._remaining_space is not None:
            self.remaining_space = self._remaining_space
        else:
            self.remaining_space = self._capacity

        if self._product_list is not None:
            for id in self._product_list:
                self.product_list.append(id)

        if self._product_number is not None:
            for n in self._product_number:
                self.product_number.append(n)
