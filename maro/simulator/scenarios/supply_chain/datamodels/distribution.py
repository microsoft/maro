# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .base import DataModelBase


@node("distribution")
class DistributionDataModel(DataModelBase):
    """Distribution data model for distribution unit."""

    remaining_order_quantity = NodeAttribute(AttributeType.UInt)
    remaining_order_number = NodeAttribute(AttributeType.UInt)

    def __init__(self) -> None:
        super(DistributionDataModel, self).__init__()

    def reset(self) -> None:
        super(DistributionDataModel, self).reset()
