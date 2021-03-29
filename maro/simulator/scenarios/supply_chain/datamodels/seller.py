# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("seller")
class SellerDataModel(SkuDataModel):
    """Data model for seller unit."""
    total_sold = NodeAttribute(AttributeType.UInt)

    backlog_ratio = NodeAttribute(AttributeType.Float)

    unit_price = NodeAttribute(AttributeType.UInt)

    sale_gamma = NodeAttribute(AttributeType.UInt)

    demand = NodeAttribute(AttributeType.UInt)
    sold = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(SellerDataModel, self).__init__()

        self._unit_price = 0
        self._backlog_ratio = 0

    def initialize(self, unit_price: int = 0, backlog_ratio: int = 0):
        self._unit_price = unit_price
        self._backlog_ratio = backlog_ratio

        self.reset()

    def reset(self):
        super(SellerDataModel, self).reset()

        self.unit_price = self._unit_price
        self.backlog_ratio = self._backlog_ratio
