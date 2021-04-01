# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("distribution")
class DistributionDataModel(DataModelBase):
    """Distribution data model for distribution unit."""
    delay_order_penalty = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    remaining_order_quantity = NodeAttribute(AttributeType.UInt)
    remaining_order_number = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(DistributionDataModel, self).__init__()

        self._unit_price = 0

    def reset(self):
        super(DistributionDataModel, self).reset()
