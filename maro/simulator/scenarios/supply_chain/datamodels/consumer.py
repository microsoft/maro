# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("consumer")
class ConsumerDataModel(ExtendDataModel):
    """Data model for consumer unit."""
    total_purchased = NodeAttribute(AttributeType.UInt)
    total_received = NodeAttribute(AttributeType.UInt)

    purchased = NodeAttribute(AttributeType.UInt)
    received = NodeAttribute(AttributeType.UInt)
    order_product_cost = NodeAttribute(AttributeType.UInt)

    latest_consumptions = NodeAttribute(AttributeType.Float)

    order_quantity = NodeAttribute(AttributeType.UInt)

    price = NodeAttribute(AttributeType.Float)
    order_cost = NodeAttribute(AttributeType.Float)

    reward_discount = NodeAttribute(AttributeType.Float)

    def __init__(self) -> None:
        super(ConsumerDataModel, self).__init__()

        self._price = 0
        self._order_cost = 0

    def initialize(self, price: int, order_cost: int) -> None:
        self._price = price
        self._order_cost = order_cost

        self.reset()

    def reset(self) -> None:
        super(ConsumerDataModel, self).reset()

        self.price = self._price
        self.order_cost = self._order_cost
