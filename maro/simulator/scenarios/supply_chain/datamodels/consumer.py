# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("consumer")
class ConsumerDataModel(SkuDataModel):
    """Data model for consumer unit."""
    order_cost = NodeAttribute(AttributeType.UInt)
    total_purchased = NodeAttribute(AttributeType.UInt)
    total_received = NodeAttribute(AttributeType.UInt)

    source_id = NodeAttribute(AttributeType.UInt)
    quantity = NodeAttribute(AttributeType.UInt)
    vlt = NodeAttribute(AttributeType.UShort)

    # Id of upstream facilities.
    sources = NodeAttribute(AttributeType.UInt, 1, is_list=True)

    purchased = NodeAttribute(AttributeType.UInt)
    received = NodeAttribute(AttributeType.UInt)
    order_product_cost = NodeAttribute(AttributeType.UInt)

    def __init__(self):
        super(ConsumerDataModel, self).__init__()

        self._order_cost = 0

    def initialize(self, order_cost: int = 0):
        """Initialize consumer data model with order_cost from configurations.

        Args:
            order_cost (int): Order cost from configuration files.
        """
        self._order_cost = order_cost

        self.reset()

    def reset(self):
        super(ConsumerDataModel, self).reset()

        self.order_cost = self._order_cost
