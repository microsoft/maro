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

    sku_id_list = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    product_storage_index = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    # Can be changed in SellerUnit.step(), DistributionUnit.step()
    # Can be changed in DistributionUnit.post_step() and ManufactureUnit.post_step()
    # Can be changed in DistributionUnit.place_order() <- triggered by ConsumerAction
    product_quantity = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    def __init__(self) -> None:
        super(StorageDataModel, self).__init__()

        self._capacity: Optional[List[int]] = None
        self._remaining_space: Optional[List[int]] = None

        self._sku_id_list: Optional[List[int]] = None
        self._product_storage_index: Optional[List[int]] = None

        self._product_quantity: Optional[List[int]] = None

    def initialize(
        self,
        capacity: List[int],
        remaining_space: List[int],
        sku_id_list: List[int] = None,
        product_storage_index: List[int] = None,
        product_quantity: List[int] = None,
    ) -> None:
        self._capacity = capacity
        self._remaining_space = remaining_space

        self._sku_id_list = sku_id_list
        self._product_storage_index = product_storage_index

        self._product_quantity = product_quantity

        self.reset()

    def reset(self) -> None:
        super(StorageDataModel, self).reset()

        for _capacity in self._capacity:
            self.capacity.append(_capacity)

        for _remaining_space in self._remaining_space:
            self.remaining_space.append(_remaining_space)

        if self._sku_id_list is not None:
            for id in self._sku_id_list:
                self.sku_id_list.append(id)

        if self._product_storage_index is not None:
            for idx in self._product_storage_index:
                self.product_storage_index.append(idx)

        if self._product_quantity is not None:
            for quantity in self._product_quantity:
                self.product_quantity.append(quantity)
