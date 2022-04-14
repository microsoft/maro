# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .extend import ExtendDataModel


@node("consumer")
class ConsumerDataModel(ExtendDataModel):
    """Data model for consumer unit."""
    purchased = NodeAttribute(AttributeType.UInt)
    received = NodeAttribute(AttributeType.UInt)
    order_product_cost = NodeAttribute(AttributeType.UInt)

    latest_consumptions = NodeAttribute(AttributeType.Float)

    order_cost = NodeAttribute(AttributeType.Float)

    def __init__(self) -> None:
        super(ConsumerDataModel, self).__init__()

        self._order_cost = 0

    def initialize(self, order_cost: float) -> None:
        self._order_cost = order_cost

        self.reset()

    def reset(self) -> None:
        super(ConsumerDataModel, self).reset()

        self.order_cost = self._order_cost
