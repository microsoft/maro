# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("storage")
class StorageDataModel(DataModelBase):
    unit_storage_cost = NodeAttribute(AttributeType.UInt)
    remaining_space = NodeAttribute(AttributeType.UInt)
    capacity = NodeAttribute(AttributeType.UInt)

    # original is stock_levels, used to save product and its number
    product_list = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    product_number = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    def __init__(self):
        super(StorageDataModel, self).__init__()

        self._unit_storage_cost = 0
        self._capacity = 0
        self._remaining_space = None

    def initialize(self, unit_storage_cost: int = 0, capacity: int = 0, remaining_space: int = None):
        self._unit_storage_cost = unit_storage_cost
        self._capacity = capacity
        self._remaining_space = remaining_space

        self.reset()

    def reset(self):
        super(StorageDataModel, self).reset()

        self.unit_storage_cost = self._unit_storage_cost
        self.capacity = self._capacity

        if self._remaining_space is not None:
            self.remaining_space = self._remaining_space
        else:
            self.remaining_space = self._capacity
