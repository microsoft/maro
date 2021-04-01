# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("consumer")
class ConsumerDataModel(SkuDataModel):
    """Data model for consumer unit."""
    total_purchased = NodeAttribute(AttributeType.UInt)
    total_received = NodeAttribute(AttributeType.UInt)

    purchased = NodeAttribute(AttributeType.UInt)
    received = NodeAttribute(AttributeType.UInt)
    order_product_cost = NodeAttribute(AttributeType.UInt)

    latest_consumptions = NodeAttribute(AttributeType.Float)
    pending_order_daily = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(ConsumerDataModel, self).__init__()

    def reset(self):
        super(ConsumerDataModel, self).reset()
