# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("distribution")
class DistributionDataModel(DataModelBase):
    """Distribution data model for distribution unit."""
    unit_price = NodeAttribute(AttributeType.Int)

    product_list = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    check_in_price = NodeAttribute(AttributeType.UInt, 1, is_list=True)
    delay_order_penalty = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    def __init__(self):
        super(DistributionDataModel, self).__init__()

        self._unit_price = 0

    def initialize(self, unit_price: int = 0):
        """Initialize distribution with unit_price from configuration.

        Args:
            unit_price (int): Unit price from configuration files.
        """
        self._unit_price = unit_price

        self.reset()

    def reset(self):
        super(DistributionDataModel, self).reset()

        self.unit_price = self._unit_price
