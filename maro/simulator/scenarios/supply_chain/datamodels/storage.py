# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("storage")
class StorageDataModel(DataModelBase):
    """Data model for storage unit."""
    remaining_space = NodeAttribute(AttributeType.UInt)
    capacity = NodeAttribute(AttributeType.UInt)

    # original is , used to save product and its number
    product_list = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    product_number = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    def __init__(self) -> None:
        super(StorageDataModel, self).__init__()

        self._capacity = 0
        self._remaining_space = None
        self._product_list = None
        self._product_number = None

    def initialize(
        self,
        capacity: int = 0,
        remaining_space: int = None,
        product_list: list = None,
        product_number: list = None,
    ) -> None:
        self._capacity = capacity
        self._remaining_space = remaining_space
        self._product_list = product_list
        self._product_number = product_number

        self.reset()

    def reset(self) -> None:
        super(StorageDataModel, self).reset()

        self.capacity = self._capacity

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
