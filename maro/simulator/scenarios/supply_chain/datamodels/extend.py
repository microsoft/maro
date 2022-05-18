# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute

from .base import DataModelBase


class ExtendDataModel(DataModelBase):
    """Data model for sku related unit."""
    # Product id of this consumer belongs to.
    sku_id = NodeAttribute(AttributeType.UInt)

    # Parent unit id.
    product_unit_id = NodeAttribute(AttributeType.UInt)

    def __init__(self) -> None:
        super(ExtendDataModel, self).__init__()

        self._sku_id = 0
        self._product_unit_id = 0

    def reset(self) -> None:
        super(ExtendDataModel, self).reset()

        self.sku_id = self._sku_id
        self.product_unit_id = self._product_unit_id

    def set_sku_id(self, sku_id: int, product_unit_id: int) -> None:
        self._sku_id = sku_id
        self._product_unit_id = product_unit_id
