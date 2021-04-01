# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("seller")
class SellerDataModel(SkuDataModel):
    """Data model for seller unit."""
    total_sold = NodeAttribute(AttributeType.UInt)

    demand = NodeAttribute(AttributeType.UInt)
    sold = NodeAttribute(AttributeType.UInt)
    total_demand = NodeAttribute(AttributeType.UInt)
    total_sold = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(SellerDataModel, self).__init__()

    def reset(self):
        super(SellerDataModel, self).reset()
